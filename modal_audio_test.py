"""
GUAVA Test on Modal
====================
公式test.pyをModalで実行するラッパースクリプト

使用方法:
  # Volume内容確認
  modal run modal_audio_test.py --check-only

  # self-reenactmentテスト実行
  modal run modal_audio_test.py

  # データパス指定
  modal run modal_audio_test.py --data-path /root/EHM-Tracker/output/processed_data/your_data
"""

import modal
import os

# === Modal Volume設定 ===
output_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=True)
assets_volume = modal.Volume.from_name("guava-weights")  # 正常なpklファイルがあるVolume

# === Modal Image定義 (generate_ply_modal.pyと同一) ===
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget",
        "libusb-1.0-0", "build-essential", "ninja-build",
        "clang", "llvm", "libclang-dev"
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

    # 3. Submodules Build
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/fused-ssim && pip install . --no-build-isolation"
    )

    # 4. Remaining libraries (xformers removed - incompatible with CUDA 11.8)
    .pip_install(
        "lightning==2.2.0", "roma==1.5.3", "imageio[pyav]", "imageio[ffmpeg]",
        "lmdb==1.6.2", "open3d==0.19.0", "plyfile==1.0.3", "omegaconf==2.3.0",
        "rich==14.0.0", "opencv-python-headless",
        "tyro==0.8.0", "onnxruntime-gpu==1.18", "onnx==1.16", "mediapipe==0.10.21",
        "transformers==4.37.0", "configer==1.3.1", "torchgeometry==0.1.2", "pynvml==13.0.1",
        "numpy==1.26.4", "colored"
    )

    # 5. Project Assets
    .add_local_dir("./assets", remote_path="/root/GUAVA/assets")
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
    .add_local_dir("./utils", remote_path="/root/GUAVA/utils")
    .add_local_dir("./dataset", remote_path="/root/GUAVA/dataset")
    .add_local_dir("./configs", remote_path="/root/GUAVA/configs")
)

app = modal.App("guava-test")


@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
        "/root/assets_volume": assets_volume,  # 正常なFLAME/SMPLXファイル
    },
    env={
        "MEDIAPIPE_DISABLE_GPU": "1",
        "XFORMERS_DISABLED": "1",  # xformersを無効化してnative attentionを使用
    }
)
def run_test(data_path: str = None, skip_self_act: bool = False):
    """
    公式test.pyのself-reenactmentテストを実行
    """
    import sys
    import glob
    import re
    import copy
    import json
    import time

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    # xformersを無効化（CUDA 11.8と互換性がないため）
    # DINOv2がxformersを検出しても使用しないようにする
    sys.modules['xformers'] = None
    sys.modules['xformers.ops'] = None

    import torch
    import numpy as np
    import imageio
    import lightning
    from tqdm import tqdm
    from omegaconf import OmegaConf

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs, to8b

    # YAMLファイルのパス修正 (generate_ply_modal.pyと同一)
    for yaml_file in glob.glob("assets/GUAVA/*.yaml"):
        with open(yaml_file, 'r') as f:
            content = f.read()
        if "C:\\" in content or "C:/" in content:
            new_content = re.sub(r'[A-Z]:[\\/].*[\\/]assets', '/root/GUAVA/assets', content)
            with open(yaml_file, 'w') as f:
                f.write(new_content)

    # Volumeから正常なassetsをコピー（Windowsアップロードで壊れたpklを上書き）
    import shutil
    volume_assets = "/root/assets_volume"
    local_assets = "/root/GUAVA/assets"
    for subdir in ["FLAME", "SMPLX", "GUAVA"]:
        src = os.path.join(volume_assets, subdir)
        dst = os.path.join(local_assets, subdir)
        if os.path.exists(src):
            print(f"Copying {subdir} from Volume...")
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # データパスの検出 (modal_final_clean.pyと同一パターン)
    if data_path is None:
        search_path = "/root/EHM-Tracker/output/processed_data"
        if os.path.exists(search_path):
            subdirs = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
            if subdirs:
                data_path = os.path.join(search_path, subdirs[0])
                print(f"自動検出されたデータパス: {data_path}")
            else:
                return {"error": f"トラッキングデータが見つかりません: {search_path}"}
        else:
            return {"error": f"EHM結果ディレクトリが見つかりません: {search_path}"}

    print(f"=== GUAVA Test (Self-Reenactment) ===")
    print(f"データパス: {data_path}")

    # モデル設定 (generate_ply_modal.pyと同一)
    model_path = "assets/GUAVA"
    model_config_path = os.path.join(model_path, 'config.yaml')

    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)
    device = 'cuda:0'

    # モデル読み込み (test.pyと同一)
    print("モデルを読み込み中...")
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    render_model.eval()

    # チェックポイント読み込み (test.pyと同一)
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
    # ConfigDictは dict と _dot_config (OmegaConf) の両方を持つ
    # データローダーは cfg.DATASET.data_path (ドットアクセス) を使うので両方更新
    meta_cfg['DATASET']['data_path'] = data_path
    # _dot_configも更新
    OmegaConf.set_readonly(meta_cfg._dot_config, False)
    meta_cfg._dot_config.DATASET.data_path = data_path
    OmegaConf.set_readonly(meta_cfg._dot_config, True)

    # デバッグ: パスと必要ファイルの存在確認
    print(f"データパス (dict): {meta_cfg['DATASET']['data_path']}")
    print(f"データパス (OmegaConf): {meta_cfg._dot_config.DATASET.data_path}")
    pkl_path = os.path.join(data_path, 'optim_tracking_ehm.pkl')
    print(f"PKLファイル: {pkl_path}")
    print(f"PKLファイル存在: {os.path.exists(pkl_path)}")
    if os.path.exists(data_path):
        print(f"data_path内容: {os.listdir(data_path)}")

    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    print(f"データセット読み込み完了: {len(test_dataset)} サンプル")

    # 出力ディレクトリ
    save_path = "/root/EHM-Tracker/output/test_results"
    os.makedirs(save_path, exist_ok=True)

    results = []
    bg = 0.0
    video_ids = list(test_dataset.videos_info.keys())

    with torch.no_grad():
        for vidx, video_id in enumerate(video_ids):
            print(f'\n{video_id} [{vidx+1}/{len(video_ids)}]')
            out_videoid_dir = os.path.join(save_path, video_id)
            out_render_path = os.path.join(out_videoid_dir, 'render')
            out_gt_path = os.path.join(out_videoid_dir, 'gt')
            os.makedirs(out_render_path, exist_ok=True)
            os.makedirs(out_gt_path, exist_ok=True)

            speed_info = {}
            source_info = test_dataset._load_source_info(video_id)

            # warmup
            vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)
            start_time = time.time()
            vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)
            infer_time = time.time() - start_time

            ubody_gaussians = Ubody_Gaussian(meta_cfg.MODEL, vertex_gs_dict, up_point_gs_dict, pruning=True)
            ubody_gaussians.init_ehm(infer_model.ehm)
            ubody_gaussians.eval()

            # PLY保存 (test.pyと同一)
            ubody_gaussians.save_point_ply(out_videoid_dir)
            ubody_gaussians.save_gaussian_ply(out_videoid_dir)

            frames = test_dataset.videos_info[video_id]['frames_keys']
            all_render_time = 0.0
            test_num = test_dataset.testing_split[video_id]
            rendering_imgs = []

            # warmup render
            target_info = test_dataset._load_target_info(video_id, frames[0])
            deform_gaussian_assets = ubody_gaussians(target_info)
            render_results = render_model(deform_gaussian_assets, target_info['render_cam_params'], bg=bg)

            for idx, frame in tqdm(enumerate(frames[-test_num:])):
                target_info = test_dataset._load_target_info(video_id, frame)
                start_time = time.time()
                deform_gaussian_assets = ubody_gaussians(target_info)
                render_results = render_model(deform_gaussian_assets, target_info['render_cam_params'], bg=bg)
                render_time = time.time() - start_time
                all_render_time += render_time

                render_image = render_results['renders'][0]
                gt_mask = target_info['mask'][0]
                gt_image = target_info['image'][0] * gt_mask + (1 - gt_mask) * bg

                import torchvision
                torchvision.utils.save_image(gt_image, os.path.join(out_gt_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(render_image, os.path.join(out_render_path, '{0:05d}'.format(idx) + ".png"))

                cat_image = torch.cat([gt_image, render_image], dim=2)
                rendering_imgs.append(to8b(cat_image.detach().cpu().numpy()))

            rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
            imageio.mimwrite(os.path.join(out_videoid_dir, f'{video_id}_video.mp4'), rendering_imgs, fps=30, quality=8)

            render_speed = test_num / all_render_time
            speed_info['infer_time (ms)'] = infer_time * 1000
            speed_info['render_speed (fps)'] = render_speed

            with open(os.path.join(out_videoid_dir, 'speed_info.json'), 'w') as f:
                json.dump(speed_info, f)

            results.append({
                'video_id': video_id,
                'infer_time_ms': infer_time * 1000,
                'render_fps': render_speed,
                'test_frames': test_num
            })

            print(f"  Infer time: {infer_time*1000:.2f}ms")
            print(f"  Render speed: {render_speed:.2f} fps")

    test_dataset._lmdb_engine.close()

    # Volume commit
    output_volume.commit()

    return {
        "success": True,
        "save_path": save_path,
        "results": results
    }


@app.function(
    image=image,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
        "/root/assets_volume": assets_volume,
    },
    timeout=300,
)
def check_volume():
    """Volumeの内容を確認"""
    import subprocess

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
            items = sorted(os.listdir(path))[:20]
            for item in items:
                full_path = os.path.join(path, item)
                contents[item] = scan_dir(full_path, depth + 1, max_depth)
            total = len(os.listdir(path))
            if total > 20:
                contents["..."] = f"and {total - 20} more items"
        except Exception as e:
            contents["ERROR"] = str(e)
        return contents

    result = {
        "ehm_volume": scan_dir("/root/EHM-Tracker/output"),
        "assets_volume": scan_dir("/root/assets_volume", max_depth=2),
        "guava_assets_local": scan_dir("/root/GUAVA/assets", max_depth=2),
    }
    return result


@app.function(
    image=image,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
    },
    timeout=300,
)
def inspect_pkl_structure():
    """optim_tracking_ehm.pklの構造を確認"""
    import pickle

    pkl_path = "/root/EHM-Tracker/output/processed_data/driving/optim_tracking_ehm.pkl"
    if not os.path.exists(pkl_path):
        # 検索
        search_path = "/root/EHM-Tracker/output/processed_data"
        for root, dirs, files in os.walk(search_path):
            if "optim_tracking_ehm.pkl" in files:
                pkl_path = os.path.join(root, "optim_tracking_ehm.pkl")
                break

    if not os.path.exists(pkl_path):
        return {"error": f"pkl not found"}

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    def get_structure(obj, depth=0, max_depth=4):
        if depth > max_depth:
            return "..."
        if isinstance(obj, dict):
            result = {}
            for k, v in list(obj.items())[:5]:  # 最初の5キーのみ
                result[str(k)] = get_structure(v, depth + 1, max_depth)
            if len(obj) > 5:
                result["..."] = f"and {len(obj) - 5} more keys"
            return result
        elif hasattr(obj, 'shape'):
            return f"array shape={obj.shape} dtype={obj.dtype}"
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 0:
                return f"list[{len(obj)}] first={get_structure(obj[0], depth + 1, max_depth)}"
            return "empty list"
        else:
            return f"{type(obj).__name__}: {str(obj)[:50]}"

    return {
        "pkl_path": pkl_path,
        "structure": get_structure(data)
    }


@app.function(
    image=image,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
    },
    timeout=300,
)
def download_results():
    """テスト結果をダウンロード用に取得"""
    import base64
    results_path = "/root/EHM-Tracker/output/test_results"

    if not os.path.exists(results_path):
        return {"error": f"結果が見つかりません: {results_path}"}

    files = {}
    for root, dirs, filenames in os.walk(results_path):
        for filename in filenames:
            if filename.endswith(('.mp4', '.json', '.png', '.ply')):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, results_path)

                # 動画とPLYはパスのみ（大きいので）
                if filename.endswith(('.mp4', '.ply')):
                    files[rel_path] = {
                        "type": "large_file",
                        "size": os.path.getsize(full_path),
                        "path": full_path
                    }
                # JSONは内容を含める
                elif filename.endswith('.json'):
                    with open(full_path, 'r') as f:
                        files[rel_path] = {
                            "type": "json",
                            "content": f.read()
                        }
                # 最初の数枚のPNGのみBase64エンコード
                elif filename.endswith('.png') and '00000' in filename:
                    with open(full_path, 'rb') as f:
                        files[rel_path] = {
                            "type": "image",
                            "content": base64.b64encode(f.read()).decode('utf-8')
                        }

    return {
        "success": True,
        "files": files,
        "results_path": results_path
    }


@app.function(
    image=image,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
    },
    timeout=600,
)
def get_video_bytes(video_name: str = None):
    """動画ファイルのバイナリを取得"""
    import base64
    results_path = "/root/EHM-Tracker/output/test_results"

    # 動画ファイルを検索
    videos = []
    for root, dirs, filenames in os.walk(results_path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                videos.append(os.path.join(root, filename))

    if not videos:
        return {"error": "動画ファイルが見つかりません"}

    # 最初の動画または指定された動画を取得
    video_path = videos[0]
    if video_name:
        for v in videos:
            if video_name in v:
                video_path = v
                break

    with open(video_path, 'rb') as f:
        video_data = f.read()

    return {
        "success": True,
        "path": video_path,
        "size": len(video_data),
        "data": base64.b64encode(video_data).decode('utf-8')
    }


@app.local_entrypoint()
def main(
    data_path: str = None,
    check_only: bool = False,
    download: bool = False,
    get_video: bool = False,
    inspect_pkl: bool = False,
):
    """
    エントリーポイント

    使用例:
        # Volume確認
        modal run modal_audio_test.py --check-only

        # PKL構造確認
        modal run modal_audio_test.py --inspect-pkl

        # テスト実行
        modal run modal_audio_test.py

        # データパス指定
        modal run modal_audio_test.py --data-path /root/EHM-Tracker/output/processed_data/concierge

        # 結果ダウンロード
        modal run modal_audio_test.py --download

        # 動画ダウンロード
        modal run modal_audio_test.py --get-video
    """
    import json
    import base64

    if check_only:
        print("=== Volume Structure ===")
        result = check_volume.remote()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if inspect_pkl:
        print("=== PKL Structure ===")
        result = inspect_pkl_structure.remote()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if download:
        print("=== Downloading Results ===")
        result = download_results.remote()
        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"結果パス: {result['results_path']}")
        print(f"\n見つかったファイル:")
        for path, info in result['files'].items():
            if info['type'] == 'large_file':
                print(f"  {path}: {info['size'] / 1024:.1f} KB")
            elif info['type'] == 'json':
                print(f"  {path}: {info['content']}")
            elif info['type'] == 'image':
                # 画像をローカルに保存
                import os
                local_path = os.path.join('test_results', path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(base64.b64decode(info['content']))
                print(f"  {path}: 保存完了 -> {local_path}")
        return

    if get_video:
        print("=== Downloading Video ===")
        result = get_video_bytes.remote()
        if "error" in result:
            print(f"Error: {result['error']}")
            return

        # 動画をローカルに保存
        import os
        video_filename = os.path.basename(result['path'])
        local_path = os.path.join('test_results', video_filename)
        os.makedirs('test_results', exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(base64.b64decode(result['data']))
        print(f"動画保存完了: {local_path} ({result['size'] / 1024:.1f} KB)")
        return

    print("=== Running GUAVA Test ===")
    result = run_test.remote(data_path=data_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    if result.get("success"):
        print(f"\n=== Test Complete ===")
        print(f"Save path: {result['save_path']}")
        for r in result['results']:
            print(f"\n  {r['video_id']}:")
            print(f"    Infer time: {r['infer_time_ms']:.2f}ms")
            print(f"    Render speed: {r['render_fps']:.2f} fps")
            print(f"    Test frames: {r['test_frames']}")
