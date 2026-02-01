"""
Modal Audio-Driven Avatar Test
WAVファイルからアバターを動かすテスト

使い方:
1. ローカルにWAVファイルを配置
2. modal run modal_audio_test.py --audio-path /path/to/test.wav
"""

import modal
import os

# GitHub repo URL (公式GUAVAリポジトリ)
GUAVA_REPO = "https://github.com/mirai-gpro/GUAVA_gpro.git"

# Modal Image定義 - GitHubからコードを取得
guava_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "git",
        "ffmpeg",
        "libsndfile1"
    )
    .pip_install(
        "torch==2.1.0",
        "torchvision",
        "numpy",
        "scipy",
        "opencv-python",
        "h5py",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "omegaconf",
        "lightning",
        "lmdb",
        "librosa",
        "soundfile",
    )
    .pip_install("gsplat==0.1.11")
    .pip_install("git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7")
    .run_commands(
        f"git clone {GUAVA_REPO} /root/guava",
        "cd /root/guava && git checkout main || true"
    )
)

app = modal.App("guava-audio-test")

# Volume定義
weights_volume = modal.Volume.from_name("guava-weights", create_if_missing=False)


def audio_to_flame_params(audio_path: str, fps: int = 30):
    """
    音声ファイルからFLAMEパラメータ（jaw_pose）を生成
    シンプルな音量ベースのリップシンク
    """
    import librosa
    import numpy as np

    # 音声読み込み
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    num_frames = int(duration * fps)

    # フレームごとの音量を計算
    hop_length = int(sr / fps)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # フレーム数に合わせてリサンプル
    if len(rms) < num_frames:
        rms = np.pad(rms, (0, num_frames - len(rms)), mode='constant')
    else:
        rms = rms[:num_frames]

    # 正規化 (0-1)
    rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

    # jaw_pose生成 (3次元: rotation around x, y, z)
    # 口を開ける = x軸周りの回転
    jaw_poses = []
    for amp in rms_normalized:
        jaw_open = amp * 0.4  # 最大0.4ラジアン
        jaw_pose = np.array([jaw_open, 0.0, 0.0], dtype=np.float32)
        jaw_poses.append(jaw_pose)

    return jaw_poses, num_frames, fps


@app.function(
    gpu="L4",
    image=guava_image,
    volumes={"/assets": weights_volume},
    timeout=1800,
)
def run_audio_avatar_test(audio_data: bytes, audio_filename: str):
    """
    音声データからアバターアニメーションを生成
    """
    import sys
    sys.path.insert(0, "/root/guava")

    import torch
    import numpy as np
    import imageio
    import copy
    from tqdm import tqdm
    from omegaconf import OmegaConf
    import lightning

    os.chdir("/root/guava")

    # 音声ファイルを一時保存
    audio_path = f"/tmp/{audio_filename}"
    with open(audio_path, "wb") as f:
        f.write(audio_data)

    print(f"Audio file saved: {audio_path}")

    # Volumeの内容を確認
    print("\n=== Checking Volume Contents ===")
    assets_path = "/assets"
    if os.path.exists(assets_path):
        for item in os.listdir(assets_path):
            print(f"  {item}")
            sub_path = os.path.join(assets_path, item)
            if os.path.isdir(sub_path):
                for sub_item in os.listdir(sub_path)[:5]:
                    print(f"    {sub_item}")

    # モデルパスとデータパスを確認
    model_path = "/assets/assets/GUAVA"
    data_base = "/assets/assets/example/tracked_video"

    if not os.path.exists(model_path):
        # 別のパス構造を試す
        if os.path.exists("/assets/GUAVA"):
            model_path = "/assets/GUAVA"
        else:
            return {"error": f"Model not found. Checked: {model_path}"}

    print(f"\nModel path: {model_path}")
    print(f"Model contents: {os.listdir(model_path)}")

    # データパスを探す
    data_path = None
    possible_data_paths = [
        "/assets/assets/example/tracked_video",
        "/assets/example/tracked_video",
        "/assets/tracked_video",
    ]

    for p in possible_data_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        return {"error": "Tracked video data not found"}

    print(f"\nData path: {data_path}")
    print(f"Available videos: {os.listdir(data_path)}")

    # 最初のビデオを使用
    video_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not video_dirs:
        return {"error": "No video directories found"}

    video_id = video_dirs[0]
    video_data_path = os.path.join(data_path, video_id)
    print(f"\nUsing video: {video_id}")

    # 音声からFLAMEパラメータ生成
    print("\n=== Generating FLAME params from audio ===")
    jaw_poses, num_frames, fps = audio_to_flame_params(audio_path)
    print(f"Generated {num_frames} frames at {fps} FPS")

    # GUAVAモデルをロード
    print("\n=== Loading GUAVA model ===")
    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs
    from utils.general_utils import to8b

    # Config
    model_config_path = os.path.join(model_path, 'config.yaml')
    if not os.path.exists(model_config_path):
        model_config_path = 'configs/train/ubody_512.yaml'

    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)
    device = 'cuda:0'

    # モデル初期化
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    infer_model.eval()
    render_model.eval()

    # チェックポイントをロード
    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best')
    if base_model is None:
        base_model = find_pt_file(ckpt_path, 'latest')

    if base_model is None:
        return {"error": f"No checkpoint found in {ckpt_path}"}

    print(f"Loading checkpoint: {base_model}")
    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    render_model.load_state_dict(_state['render_model'], strict=False)

    # データセットをロード
    OmegaConf.set_readonly(meta_cfg['DATASET'], False)
    meta_cfg['DATASET']['data_path'] = video_data_path

    print(f"\n=== Loading dataset from {video_data_path} ===")
    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)

    # ソース情報をロード（アバターの外見）
    video_ids = list(test_dataset.videos_info.keys())
    if not video_ids:
        return {"error": "No video IDs found in dataset"}

    source_video_id = video_ids[0]
    print(f"Source video ID: {source_video_id}")

    source_info = test_dataset._load_source_info(source_video_id)

    # アバター生成
    print("\n=== Generating Avatar ===")
    with torch.no_grad():
        vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)
        ubody_gaussians = Ubody_Gaussian(meta_cfg.MODEL, vertex_gs_dict, up_point_gs_dict, pruning=True)
        ubody_gaussians.init_ehm(infer_model.ehm)
        ubody_gaussians.eval()

    # ベースとなるターゲット情報を取得
    frames_keys = test_dataset.videos_info[source_video_id]['frames_keys']
    base_target_info = test_dataset._load_target_info(source_video_id, frames_keys[0])

    # 音声に合わせてレンダリング
    print("\n=== Rendering with audio ===")
    rendering_imgs = []
    bg = 0.0

    with torch.no_grad():
        for frame_idx in tqdm(range(min(num_frames, 300))):  # 最大300フレーム
            # ターゲット情報をコピー
            target_info = copy.deepcopy(base_target_info)

            # jaw_poseを更新（音声から生成したもの）
            jaw_pose = torch.tensor(jaw_poses[frame_idx], dtype=torch.float32, device=device)
            target_info['flame_coeffs']['jaw_pose'] = jaw_pose.unsqueeze(0)

            # レンダリング
            deform_gaussian_assets = ubody_gaussians(target_info)
            render_results = render_model(deform_gaussian_assets, target_info['render_cam_params'], bg=bg)

            render_image = render_results['renders'][0]
            rendering_imgs.append(to8b(render_image.detach().cpu().numpy()))

    # 動画保存
    print("\n=== Saving video ===")
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)

    output_path = "/tmp/audio_avatar_output.mp4"
    imageio.mimwrite(output_path, rendering_imgs, fps=fps, quality=8)

    # 動画をバイトで返す
    with open(output_path, "rb") as f:
        video_data = f.read()

    test_dataset._lmdb_engine.close()

    return {
        "success": True,
        "num_frames": len(rendering_imgs),
        "fps": fps,
        "video_data": video_data,
    }


@app.function(
    image=guava_image,
    volumes={"/assets": weights_volume},
    timeout=600,
)
def check_volume_structure():
    """Volumeの内容を確認"""
    import os

    result = {"structure": {}}

    def scan_dir(path, depth=0, max_depth=3):
        if depth > max_depth:
            return "..."

        if not os.path.exists(path):
            return "NOT EXISTS"

        if os.path.isfile(path):
            size = os.path.getsize(path)
            return f"FILE ({size} bytes)"

        contents = {}
        try:
            items = os.listdir(path)[:20]  # 最大20項目
            for item in items:
                full_path = os.path.join(path, item)
                contents[item] = scan_dir(full_path, depth + 1, max_depth)
            if len(os.listdir(path)) > 20:
                contents["..."] = f"and {len(os.listdir(path)) - 20} more items"
        except Exception as e:
            contents["ERROR"] = str(e)

        return contents

    result["structure"] = scan_dir("/assets")
    return result


@app.local_entrypoint()
def main(
    audio_path: str = None,
    check_only: bool = False,
):
    """
    メインエントリーポイント

    使い方:
        # Volumeの内容を確認
        modal run modal_audio_test.py --check-only

        # 音声ファイルでテスト
        modal run modal_audio_test.py --audio-path ./test.wav
    """
    import json

    if check_only:
        print("Checking volume structure...")
        result = check_volume_structure.remote()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if audio_path is None:
        print("Error: --audio-path is required")
        print("Usage: modal run modal_audio_test.py --audio-path /path/to/test.wav")
        return

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    # 音声ファイルを読み込み
    print(f"Loading audio file: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    audio_filename = os.path.basename(audio_path)

    # Modalで実行
    print("Running on Modal...")
    result = run_audio_avatar_test.remote(audio_data, audio_filename)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    if result.get("success"):
        # 動画を保存
        output_path = f"./output_avatar_{audio_filename.replace('.wav', '.mp4')}"
        with open(output_path, "wb") as f:
            f.write(result["video_data"])

        print(f"\nSuccess!")
        print(f"  Frames: {result['num_frames']}")
        print(f"  FPS: {result['fps']}")
        print(f"  Output: {output_path}")
    else:
        print(f"Result: {result}")
