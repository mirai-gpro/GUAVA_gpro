"""
Modal Audio-Driven Avatar Test
WAVファイルからアバターを動かすテスト

使い方:
1. ローカルにWAVファイルを配置
2. modal run modal_audio_test.py --audio-path /path/to/test.wav

Based on working generate_ply_modal.py configuration
"""

import modal
import os
import sys

# === Modal Volume設定 ===
weights_volume = modal.Volume.from_name("guava-weights", create_if_missing=True)

# === Modal Image定義 (generate_ply_modal.pyベース) ===
guava_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget",
        "libusb-1.0-0", "build-essential", "ninja-build",
        "clang", "llvm", "libclang-dev", "libsndfile1"
    )

    # 1. Base dependencies
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy<2.0'"
    )
    .run_commands(
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118"
    )

    # 2. Build Tools & Core Libraries
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "CC": "clang",
        "CXX": "clang++"
    })
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7 --no-build-isolation"
    )

    # 3. Submodules Build (ローカルからコピー)
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/fused-ssim && pip install . --no-build-isolation"
    )

    # 4. Remaining libraries
    .pip_install(
        "lightning==2.2.0", "roma==1.5.3", "imageio[pyav]", "imageio[ffmpeg]",
        "lmdb==1.6.2", "open3d==0.19.0", "plyfile==1.0.3", "omegaconf==2.3.0",
        "rich==14.0.0", "opencv-python-headless", "xformers==0.0.24",
        "tyro==0.8.0", "onnxruntime-gpu==1.18", "onnx==1.16", "mediapipe==0.10.21",
        "transformers==4.37.0", "configer==1.3.1", "torchgeometry==0.1.2", "pynvml==13.0.1",
        "numpy==1.26.4", "colored", "librosa", "soundfile"
    )

    # 5. Project Assets (ローカルからコピー)
    .add_local_dir("./assets", remote_path="/root/GUAVA/assets")
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
    .add_local_dir("./utils", remote_path="/root/GUAVA/utils")
    .add_local_dir("./dataset", remote_path="/root/GUAVA/dataset")
    .add_local_dir("./configs", remote_path="/root/GUAVA/configs")
)

app = modal.App("guava-audio-test")


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
    gpu="a10g",
    image=guava_image,
    volumes={"/root/EHM_results": weights_volume},
    timeout=1800,
    env={"MEDIAPIPE_DISABLE_GPU": "1"}
)
def run_audio_avatar_test(audio_data: bytes, audio_filename: str, data_path: str = None):
    """
    音声データからアバターアニメーションを生成
    """
    import glob
    import re
    import copy

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    import torch
    import numpy as np
    import imageio
    from tqdm import tqdm
    from omegaconf import OmegaConf
    import lightning

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs, to8b

    # 音声ファイルを一時保存
    audio_path = f"/tmp/{audio_filename}"
    with open(audio_path, "wb") as f:
        f.write(audio_data)

    print(f"Audio file saved: {audio_path}")

    # YAMLファイルのパス修正
    for yaml_file in glob.glob("assets/GUAVA/*.yaml"):
        with open(yaml_file, 'r') as f:
            content = f.read()
        if "C:\\" in content or "C:/" in content:
            new_content = re.sub(r'[A-Z]:[\\/].*[\\/]assets', '/root/GUAVA/assets', content)
            with open(yaml_file, 'w') as f:
                f.write(new_content)

    # データパスの検出
    if data_path is None:
        # Volumeの実際の構造に合わせたパス
        search_paths = [
            "/root/EHM_results/example/tracked_video",
            "/root/EHM_results/example/tracked_image",
            "/root/EHM_results/processed_data",
        ]
        for search_path in search_paths:
            if os.path.exists(search_path):
                subdirs = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
                if subdirs:
                    data_path = os.path.join(search_path, subdirs[0])
                    print(f"自動検出されたデータパス: {data_path}")
                    break

        if data_path is None:
            return {"error": f"トラッキングデータが見つかりません。検索パス: {search_paths}"}

    print(f"\n=== GUAVA Audio Avatar Test ===")
    print(f"データパス: {data_path}")

    # モデル設定 - Volumeからモデルを読み込む
    model_path = "/root/EHM_results/GUAVA"
    if not os.path.exists(model_path):
        # フォールバック: ローカルのassetsを使用
        model_path = "/root/GUAVA/assets/GUAVA"

    model_config_path = os.path.join(model_path, 'config.yaml')

    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)
    device = 'cuda:0'

    # モデル読み込み
    print("モデルを読み込み中...")
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    render_model.eval()

    # チェックポイント読み込み
    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best')
    if base_model is None:
        base_model = find_pt_file(ckpt_path, 'latest')

    if base_model is None or not os.path.exists(base_model):
        return {"error": f"モデルチェックポイントが見つかりません: {ckpt_path}"}

    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    render_model.load_state_dict(_state['render_model'], strict=False)
    print(f"モデル読み込み完了: {base_model}")

    # データセット設定
    OmegaConf.set_readonly(meta_cfg['DATASET'], False)
    meta_cfg['DATASET']['data_path'] = data_path

    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    print(f"データセット読み込み完了: {len(test_dataset)} サンプル")

    video_ids = list(test_dataset.videos_info.keys())
    if not video_ids:
        return {"error": "No video IDs found in dataset"}

    video_id = video_ids[0]
    print(f"Using video: {video_id}")

    # 音声からFLAMEパラメータ生成
    print("\n=== Generating FLAME params from audio ===")
    jaw_poses, num_frames, fps = audio_to_flame_params(audio_path)
    print(f"Generated {num_frames} frames at {fps} FPS")

    with torch.no_grad():
        # ソース情報読み込み
        source_info = test_dataset._load_source_info(video_id)

        # Ubody Gaussians推論
        print("Ubody Gaussians推論中...")
        vertex_gs_dict, up_point_gs_dict, extra_dict = infer_model(source_info)

        ubody_gaussians = Ubody_Gaussian(
            meta_cfg.MODEL,
            vertex_gs_dict,
            up_point_gs_dict,
            pruning=True
        )
        ubody_gaussians.init_ehm(infer_model.ehm)
        ubody_gaussians.eval()

        # ベースとなるターゲット情報を取得
        frames_keys = test_dataset.videos_info[video_id]['frames_keys']
        base_target_info = test_dataset._load_target_info(video_id, frames_keys[0])

        # 音声に合わせてレンダリング
        print("\n=== Rendering with audio ===")
        rendering_imgs = []
        bg = 0.0

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
    volumes={"/root/EHM_results": weights_volume},
    timeout=600,
    env={"MEDIAPIPE_DISABLE_GPU": "1"}
)
def check_volume_structure():
    """Volumeの内容を確認"""
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
            items = os.listdir(path)[:20]
            for item in items:
                full_path = os.path.join(path, item)
                contents[item] = scan_dir(full_path, depth + 1, max_depth)
            if len(os.listdir(path)) > 20:
                contents["..."] = f"and {len(os.listdir(path)) - 20} more items"
        except Exception as e:
            contents["ERROR"] = str(e)

        return contents

    result["structure"] = scan_dir("/root/EHM_results")
    return result


@app.local_entrypoint()
def main(
    audio_path: str = None,
    check_only: bool = False,
    data_path: str = None,
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
    result = run_audio_avatar_test.remote(audio_data, audio_filename, data_path)

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
