"""
GUAVA PLY File Generator for Modal
===================================
論文「GUAVA: Generalizable Upper Body 3D Gaussian Avatar」に準拠した
PLYファイルを生成するModalスクリプト

出力形式:
- GS_canonical.ply: 3D Gaussian Splatting形式（論文準拠）
  - 位置 (x, y, z)
  - 法線 (nx, ny, nz)
  - SH係数 (f_dc_0, f_dc_1, f_dc_2)
  - 不透明度 (opacity) - inverse sigmoid
  - スケール (scale_0, scale_1, scale_2) - log scale
  - 回転 (rot_0, rot_1, rot_2, rot_3) - quaternion (wxyz)

- canonical.ply: Open3D形式（点群+色）

使用方法:
  modal run generate_ply_modal.py
"""

import modal
import os
import sys

# === Modal Volume設定 ===
ehm_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=True)
ply_output_volume = modal.Volume.from_name("guava-ply-output", create_if_missing=True)

# === Modal Image定義 ===
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

    # 4. Remaining libraries
    .pip_install(
        "lightning==2.2.0", "roma==1.5.3", "imageio[pyav]", "imageio[ffmpeg]",
        "lmdb==1.6.2", "open3d==0.19.0", "plyfile==1.0.3", "omegaconf==2.3.0",
        "rich==14.0.0", "opencv-python-headless", "xformers==0.0.24",
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

app = modal.App("guava-ply-generator")


@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={
        "/root/EHM_results": ehm_volume,
        "/root/GUAVA/ply_outputs": ply_output_volume
    },
    env={"MEDIAPIPE_DISABLE_GPU": "1"}
)
def generate_ply(
    data_path: str = None,
    output_name: str = "avatar",
    save_split: bool = True,
    save_point_cloud: bool = True,
    save_gaussian: bool = True
):
    """
    論文準拠のPLYファイルを生成

    Args:
        data_path: EHM-Trackerでトラッキング済みのデータパス
        output_name: 出力ディレクトリ名
        save_split: SMPLX/UV別々に保存するか
        save_point_cloud: Open3D形式の点群PLYを保存するか
        save_gaussian: 3DGS形式のPLYを保存するか
    """
    import glob
    import re
    import copy

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    import torch
    import lightning
    from omegaconf import OmegaConf

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
    from utils.general_utils import ConfigDict, device_parser, calc_parameters, find_pt_file, add_extra_cfgs

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
        search_path = "/root/EHM_results/processed_data"
        if os.path.exists(search_path):
            subdirs = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
            if subdirs:
                data_path = os.path.join(search_path, subdirs[0])
                print(f"自動検出されたデータパス: {data_path}")
            else:
                raise FileNotFoundError(f"トラッキングデータが見つかりません: {search_path}")
        else:
            raise FileNotFoundError(f"EHM結果ディレクトリが見つかりません: {search_path}")

    print(f"=== GUAVA PLY Generator ===")
    print(f"データパス: {data_path}")
    print(f"出力名: {output_name}")
    print(f"分割保存: {save_split}")
    print(f"点群PLY: {save_point_cloud}")
    print(f"Gaussian PLY: {save_gaussian}")
    print("=" * 40)

    # モデル設定
    model_path = "assets/GUAVA"
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

    # チェックポイント読み込み
    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best')
    if base_model is None:
        base_model = find_pt_file(ckpt_path, 'latest')

    if base_model is None or not os.path.exists(base_model):
        raise FileNotFoundError(f"モデルチェックポイントが見つかりません: {ckpt_path}")

    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    print(f"モデル読み込み完了: {base_model}")

    # データセット設定
    OmegaConf.set_readonly(meta_cfg['DATASET'], False)
    meta_cfg['DATASET']['data_path'] = data_path

    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    print(f"データセット読み込み完了: {len(test_dataset)} サンプル")

    # 出力ディレクトリ
    output_dir = os.path.join("/root/GUAVA/ply_outputs", output_name)
    os.makedirs(output_dir, exist_ok=True)

    video_ids = list(test_dataset.videos_info.keys())

    results = []

    with torch.no_grad():
        for vidx, video_id in enumerate(video_ids):
            print(f"\n処理中: {video_id} [{vidx+1}/{len(video_ids)}]")

            video_output_dir = os.path.join(output_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)

            # ソース情報読み込み
            source_info = test_dataset._load_source_info(video_id)

            # Ubody Gaussians推論
            print("  Ubody Gaussians推論中...")
            vertex_gs_dict, up_point_gs_dict, extra_dict = infer_model(source_info)

            # Ubody Gaussian オブジェクト作成
            ubody_gaussians = Ubody_Gaussian(
                meta_cfg.MODEL,
                vertex_gs_dict,
                up_point_gs_dict,
                pruning=True
            )
            ubody_gaussians.init_ehm(infer_model.ehm)
            ubody_gaussians.eval()

            # PLYファイル保存
            ply_files = []

            if save_point_cloud:
                print("  点群PLYを保存中...")
                ubody_gaussians.save_point_ply(video_output_dir, save_split=save_split)
                ply_files.append(os.path.join(video_output_dir, 'canonical.ply'))
                if save_split:
                    ply_files.append(os.path.join(video_output_dir, 'canonical_smplx.ply'))
                    ply_files.append(os.path.join(video_output_dir, 'canonical_uv.ply'))

            if save_gaussian:
                print("  Gaussian PLYを保存中...")
                ubody_gaussians.save_gaussian_ply(video_output_dir, save_split=save_split)
                ply_files.append(os.path.join(video_output_dir, 'GS_canonical.ply'))
                if save_split:
                    ply_files.append(os.path.join(video_output_dir, 'GS_canonical_smplx.ply'))
                    ply_files.append(os.path.join(video_output_dir, 'GS_canonical_uv.ply'))

            # 統計情報を取得
            num_template_gaussians = vertex_gs_dict['positions'].shape[1]
            num_uv_gaussians = up_point_gs_dict['opacities'].shape[1]
            total_gaussians = num_template_gaussians + num_uv_gaussians

            result = {
                'video_id': video_id,
                'output_dir': video_output_dir,
                'ply_files': ply_files,
                'num_template_gaussians': num_template_gaussians,
                'num_uv_gaussians': num_uv_gaussians,
                'total_gaussians': total_gaussians
            }
            results.append(result)

            print(f"  完了: Template={num_template_gaussians}, UV={num_uv_gaussians}, Total={total_gaussians}")

    # データセットクローズ
    test_dataset._lmdb_engine.close()

    # 結果をJSONで保存
    import json
    summary = {
        'data_path': data_path,
        'output_dir': output_dir,
        'total_videos': len(results),
        'videos': results
    }

    summary_path = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== 生成完了 ===")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"処理動画数: {len(results)}")

    # Volume コミット
    ply_output_volume.commit()

    return summary


@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={
        "/root/EHM_results": ehm_volume,
        "/root/GUAVA/ply_outputs": ply_output_volume
    },
    env={"MEDIAPIPE_DISABLE_GPU": "1"}
)
def generate_ply_from_single_frame(
    data_path: str,
    frame_key: str = None,
    output_name: str = "single_frame",
    save_deformed: bool = True
):
    """
    単一フレームからPLYを生成（deformed状態も含む）

    Args:
        data_path: トラッキング済みデータパス
        frame_key: 特定フレームのキー（Noneなら最初のフレーム）
        output_name: 出力名
        save_deformed: deformed（ポーズ適用後）のPLYも保存するか
    """
    import glob
    import re
    import json

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    import torch
    import lightning
    from omegaconf import OmegaConf

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs

    # YAMLパス修正
    for yaml_file in glob.glob("assets/GUAVA/*.yaml"):
        with open(yaml_file, 'r') as f:
            content = f.read()
        if "C:\\" in content or "C:/" in content:
            new_content = re.sub(r'[A-Z]:[\\/].*[\\/]assets', '/root/GUAVA/assets', content)
            with open(yaml_file, 'w') as f:
                f.write(new_content)

    print(f"=== Single Frame PLY Generator ===")
    print(f"データパス: {data_path}")

    model_path = "assets/GUAVA"
    model_config_path = os.path.join(model_path, 'config.yaml')

    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)
    device = 'cuda:0'

    # モデル読み込み
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    render_model.eval()

    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best') or find_pt_file(ckpt_path, 'latest')

    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    render_model.load_state_dict(_state['render_model'], strict=False)

    # データセット
    OmegaConf.set_readonly(meta_cfg['DATASET'], False)
    meta_cfg['DATASET']['data_path'] = data_path

    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)

    video_ids = list(test_dataset.videos_info.keys())
    video_id = video_ids[0]

    output_dir = os.path.join("/root/GUAVA/ply_outputs", output_name)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        source_info = test_dataset._load_source_info(video_id)

        # 推論
        vertex_gs_dict, up_point_gs_dict, extra_dict = infer_model(source_info)

        ubody_gaussians = Ubody_Gaussian(
            meta_cfg.MODEL,
            vertex_gs_dict,
            up_point_gs_dict,
            pruning=True
        )
        ubody_gaussians.init_ehm(infer_model.ehm)
        ubody_gaussians.eval()

        # Canonical PLY保存
        ubody_gaussians.save_point_ply(output_dir, save_split=True)
        ubody_gaussians.save_gaussian_ply(output_dir, save_split=True)

        # Deformed PLY保存（特定フレームのポーズで変形）
        if save_deformed:
            frames = test_dataset.videos_info[video_id]['frames_keys']
            target_frame = frame_key if frame_key else frames[0]

            target_info = test_dataset._load_target_info(video_id, target_frame)
            deform_gaussian_assets = ubody_gaussians(target_info)

            # Deformed点群を保存
            ubody_gaussians.save_point_ply(output_dir, save_split=False, assets=deform_gaussian_assets)
            print(f"Deformed PLY保存完了: {target_frame}")

    test_dataset._lmdb_engine.close()

    result = {
        'video_id': video_id,
        'output_dir': output_dir,
        'frame_key': frame_key or frames[0] if save_deformed else None
    }

    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(result, f, indent=2)

    ply_output_volume.commit()

    print(f"=== 完了 ===")
    print(f"出力: {output_dir}")

    return result


@app.local_entrypoint()
def main(
    data_path: str = None,
    output_name: str = "guava_avatar",
    split: bool = True,
    gaussian_only: bool = False
):
    """
    PLY生成のエントリーポイント

    使用例:
        # 基本的な使用法（自動検出）
        modal run generate_ply_modal.py

        # データパス指定
        modal run generate_ply_modal.py --data-path /path/to/tracked/data

        # Gaussian PLYのみ
        modal run generate_ply_modal.py --gaussian-only
    """
    print("=== Modal PLY Generator 開始 ===")

    result = generate_ply.remote(
        data_path=data_path,
        output_name=output_name,
        save_split=split,
        save_point_cloud=not gaussian_only,
        save_gaussian=True
    )

    print("\n=== 生成結果 ===")
    print(f"出力ディレクトリ: {result['output_dir']}")
    print(f"処理動画数: {result['total_videos']}")

    for video in result['videos']:
        print(f"\n  {video['video_id']}:")
        print(f"    Template Gaussians: {video['num_template_gaussians']}")
        print(f"    UV Gaussians: {video['num_uv_gaussians']}")
        print(f"    Total: {video['total_gaussians']}")
        print(f"    PLYファイル:")
        for ply in video['ply_files']:
            print(f"      - {os.path.basename(ply)}")


if __name__ == "__main__":
    # ローカル実行用（テスト）
    print("このスクリプトはModalで実行してください:")
    print("  modal run generate_ply_modal.py")
    print("  modal run generate_ply_modal.py --data-path /path/to/data")
