# extract_uv_styleunet_data.py
# UV StyleUNet用のデータ抽出スクリプト
#
# 抽出対象:
#   - 入力: 35ch (3ch RGB + 32ch DINOv2) in UV space
#   - 出力: 96ch UV features
#   - Extra Style: 512-dim global feature
#
# 使用方法:
#   modal run extract_uv_styleunet_data.py --action extract --num-frames 100
#   modal run extract_uv_styleunet_data.py --action verify

import modal
import os

# Modal volumes
ehm_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=True)
weights_volume = modal.Volume.from_name("guava-weights", create_if_missing=True)
output_volume = modal.Volume.from_name("uv-styleunet-distill-data", create_if_missing=True)

# === Modal Image定義 (generate_ply_modal.py の成功パターンを完全コピー) ===
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget",
        "libusb-1.0-0", "build-essential", "ninja-build",
        "clang", "llvm", "libclang-dev"
    )

    # 1. Base dependencies - numpy<2.0 を最初にピン留め
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install 'numpy<2.0'"
    )
    .run_commands(
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118"
    )

    # 2. Build Tools & Core Libraries - 環境変数を設定してからビルド
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

    # 3. Submodules Build - generate_ply_modal.py と同じ構成
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/fused-ssim && pip install . --no-build-isolation"
    )

    # 4. Remaining libraries - generate_ply_modal.py と同じ構成
    .pip_install(
        "lightning==2.2.0", "roma==1.5.3", "imageio[pyav]", "imageio[ffmpeg]",
        "lmdb==1.6.2", "open3d==0.19.0", "plyfile==1.0.3", "omegaconf==2.3.0",
        "rich==14.0.0", "opencv-python-headless", "xformers==0.0.24",
        "tyro==0.8.0", "onnxruntime-gpu==1.18", "onnx==1.16", "mediapipe==0.10.21",
        "transformers==4.37.0", "configer==1.3.1", "torchgeometry==0.1.2", "pynvml==13.0.1",
        "einops", "easydict", "trimesh", "tqdm", "pillow", "pyyaml", "scipy", "smplx",
        "numpy==1.26.4", "colored"
    )

    # 5. Project Assets - generate_ply_modal.py と同じ構成
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
    .add_local_dir("./utils", remote_path="/root/GUAVA/utils")
    .add_local_dir("./dataset", remote_path="/root/GUAVA/dataset")
    .add_local_dir("./configs", remote_path="/root/GUAVA/configs")
)

app = modal.App("uv-styleunet-data-extraction")


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 3600,
    volumes={
        "/root/EHM_results": ehm_volume,
        "/root/GUAVA/assets": weights_volume,
        "/root/distill_data": output_volume,
    },
)
def extract_uv_styleunet_data(num_frames: int = 100, num_angles: int = 9):
    """
    UV StyleUNet (uv_feature_decoder) の入出力データを抽出

    Hook対象: infer_model.uv_feature_decoder
    - 入力: 35ch (3ch RGB + 32ch DINOv2) @ UV空間
    - 出力: 96ch UV features
    - Extra Style: 512-dim (uv_style_mapping出力)
    """
    import sys
    import os

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    import torch
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    from easydict import EasyDict
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = Path("/root/distill_data/uv_styleunet_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_35ch").mkdir(exist_ok=True)
    (output_dir / "output_96ch").mkdir(exist_ok=True)
    (output_dir / "extra_style").mkdir(exist_ok=True)

    # Load config
    config_path = "/root/GUAVA/assets/GUAVA/config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    meta_cfg = EasyDict(cfg)
    print(f"Config loaded: {meta_cfg.MODEL}")

    # Load checkpoint
    ckpt_path = "/root/GUAVA/assets/GUAVA/checkpoints/best_160000.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    print(f"Checkpoint keys: {list(state.keys())}")

    # Import and create model
    from models.UbodyAvatar.ubody_gaussian import Ubody_Gaussian_inferer

    print("Creating Ubody_Gaussian_inferer...")
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL).to(device)

    # Load weights
    model_state = state.get('model', state)

    # Filter keys for infer_model
    infer_keys = {k: v for k, v in model_state.items() if not k.startswith('render_model.')}
    missing, unexpected = infer_model.load_state_dict(infer_keys, strict=False)
    print(f"Loaded infer_model weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    infer_model.eval()

    # ============================================================
    # Hook Setup for UV StyleUNet (uv_feature_decoder)
    # ============================================================

    captured_data = {
        'input': None,      # 35ch input
        'output': None,     # 96ch output
        'extra_style': None # 512-dim style
    }

    def hook_uv_feature_decoder(module, input, output):
        """Capture UV StyleUNet input/output"""
        # input is tuple: (x, extra_style) or just x
        if isinstance(input, tuple):
            x = input[0]
            # extra_style might be passed as keyword arg
        else:
            x = input

        captured_data['input'] = x.detach().cpu()
        captured_data['output'] = output.detach().cpu()

    # Also hook uv_style_mapping to capture extra_style
    original_uv_style_mapping_forward = infer_model.uv_style_mapping.forward

    def hooked_uv_style_mapping_forward(x):
        result = original_uv_style_mapping_forward(x)
        captured_data['extra_style'] = result.detach().cpu()
        return result

    infer_model.uv_style_mapping.forward = hooked_uv_style_mapping_forward

    # Register hook on uv_feature_decoder
    hook_handle = infer_model.uv_feature_decoder.register_forward_hook(hook_uv_feature_decoder)

    print("Hooks registered on uv_feature_decoder and uv_style_mapping")

    # ============================================================
    # Load EHM Tracking Data (generate_ply_cloud.py と同じ方法)
    # ============================================================

    from omegaconf import OmegaConf
    from dataset import TrackedData_infer
    from utils.general_utils import ConfigDict, add_extra_cfgs

    # find_tracking_data 関数 (generate_ply_cloud.py からコピー)
    def find_tracking_data(base_path, max_depth=3):
        if max_depth <= 0:
            return None
        tracking_file = os.path.join(base_path, 'optim_tracking_ehm.pkl')
        if os.path.exists(tracking_file):
            return base_path
        if os.path.isdir(base_path):
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    result = find_tracking_data(item_path, max_depth - 1)
                    if result:
                        return result
        return None

    search_path = "/root/EHM_results/processed_data"
    data_path = None

    if os.path.exists(search_path):
        data_path = find_tracking_data(search_path)
        if data_path:
            print(f"Found tracking data: {data_path}")
        else:
            print(f"No optim_tracking_ehm.pkl found in {search_path}")
            return {"error": "No tracking data"}
    else:
        print(f"EHM results directory not found: {search_path}")
        return {"error": "No tracking data"}

    # Dataset設定
    model_path = "assets/GUAVA"
    model_config_path = os.path.join(model_path, 'config.yaml')
    meta_cfg_dataset = ConfigDict(model_config_path=model_config_path)
    meta_cfg_dataset = add_extra_cfgs(meta_cfg_dataset)

    meta_cfg_dataset['DATASET']['data_path'] = data_path
    OmegaConf.set_readonly(meta_cfg_dataset._dot_config, False)
    meta_cfg_dataset._dot_config.DATASET.data_path = data_path
    OmegaConf.set_readonly(meta_cfg_dataset._dot_config, True)

    test_dataset = TrackedData_infer(cfg=meta_cfg_dataset, split='test', device=device, test_full=True)
    print(f"Dataset loaded: {len(test_dataset)} samples")

    video_ids = list(test_dataset.videos_info.keys())
    print(f"Found {len(video_ids)} videos")

    # ============================================================
    # Data Extraction Loop
    # ============================================================

    sample_count = 0

    for video_id in video_ids:
        if sample_count >= num_frames:
            break

        print(f"\nProcessing video: {video_id}")

        try:
            # Load source info (generate_ply_cloud.py と同じ方法)
            source_info = test_dataset._load_source_info(video_id)

            # Forward pass (triggers hooks)
            with torch.no_grad():
                vertex_gs_dict, uv_point_gs_dict, extra_dict = infer_model(source_info)

            # Check if data was captured
            if captured_data['input'] is None or captured_data['output'] is None:
                print(f"  No data captured for video {video_id}")
                continue

            # Save captured data
            sample_id = f"{sample_count:06d}"

            torch.save(captured_data['input'], output_dir / "input_35ch" / f"{sample_id}.pt")
            torch.save(captured_data['output'], output_dir / "output_96ch" / f"{sample_id}.pt")

            if captured_data['extra_style'] is not None:
                torch.save(captured_data['extra_style'], output_dir / "extra_style" / f"{sample_id}.pt")

            print(f"  Saved sample {sample_id}")
            print(f"    Input shape: {captured_data['input'].shape if captured_data['input'] is not None else 'None'}")
            print(f"    Output shape: {captured_data['output'].shape if captured_data['output'] is not None else 'None'}")

            sample_count += 1

            # Reset captured data
            captured_data['input'] = None
            captured_data['output'] = None
            captured_data['extra_style'] = None

        except Exception as e:
            print(f"  Error processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    test_dataset._lmdb_engine.close()

    # Cleanup
    hook_handle.remove()

    # Summary
    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {sample_count}")
    print(f"Output directory: {output_dir}")
    print(f"  - input_35ch/: {len(list((output_dir / 'input_35ch').glob('*.pt')))} files")
    print(f"  - output_96ch/: {len(list((output_dir / 'output_96ch').glob('*.pt')))} files")
    print(f"  - extra_style/: {len(list((output_dir / 'extra_style').glob('*.pt')))} files")

    # Verify sample shapes
    if sample_count > 0:
        sample_input = torch.load(output_dir / "input_35ch" / "000000.pt")
        sample_output = torch.load(output_dir / "output_96ch" / "000000.pt")
        print(f"\nSample shapes:")
        print(f"  Input: {sample_input.shape} (expected: [1, 35, 512, 512])")
        print(f"  Output: {sample_output.shape} (expected: [1, 96, 512, 512])")

        if (output_dir / "extra_style" / "000000.pt").exists():
            sample_style = torch.load(output_dir / "extra_style" / "000000.pt")
            print(f"  Extra Style: {sample_style.shape} (expected: [1, 512])")

    output_volume.commit()

    return {
        "total_samples": sample_count,
        "output_dir": str(output_dir),
    }


@app.function(
    image=image,
    timeout=600,
    volumes={"/root/distill_data": output_volume},
)
def verify_extracted_data():
    """抽出データの検証"""
    import torch
    from pathlib import Path

    output_dir = Path("/root/distill_data/uv_styleunet_dataset")

    if not output_dir.exists():
        return {"error": "No data found"}

    input_files = sorted((output_dir / "input_35ch").glob("*.pt"))
    output_files = sorted((output_dir / "output_96ch").glob("*.pt"))
    style_files = sorted((output_dir / "extra_style").glob("*.pt"))

    print(f"Found {len(input_files)} input files")
    print(f"Found {len(output_files)} output files")
    print(f"Found {len(style_files)} style files")

    # Verify shapes
    if input_files:
        sample_input = torch.load(input_files[0])
        sample_output = torch.load(output_files[0])

        print(f"\nSample shapes:")
        print(f"  Input: {sample_input.shape}")
        print(f"  Output: {sample_output.shape}")

        # Check values
        print(f"\nInput stats:")
        print(f"  Min: {sample_input.min():.4f}, Max: {sample_input.max():.4f}")
        print(f"  Mean: {sample_input.mean():.4f}, Std: {sample_input.std():.4f}")

        print(f"\nOutput stats:")
        print(f"  Min: {sample_output.min():.4f}, Max: {sample_output.max():.4f}")
        print(f"  Mean: {sample_output.mean():.4f}, Std: {sample_output.std():.4f}")

        if style_files:
            sample_style = torch.load(style_files[0])
            print(f"\nStyle stats:")
            print(f"  Shape: {sample_style.shape}")
            print(f"  Min: {sample_style.min():.4f}, Max: {sample_style.max():.4f}")

    return {
        "input_files": len(input_files),
        "output_files": len(output_files),
        "style_files": len(style_files),
    }


@app.local_entrypoint()
def main(action: str = "extract", num_frames: int = 100, num_angles: int = 9):
    """
    UV StyleUNet データ抽出

    使用方法:
        # データ抽出
        modal run extract_uv_styleunet_data.py --action extract --num-frames 100 --num-angles 9

        # データ検証
        modal run extract_uv_styleunet_data.py --action verify
    """
    if action == "extract":
        print(f"Extracting UV StyleUNet data: {num_frames} frames, {num_angles} angles per frame")
        result = extract_uv_styleunet_data.remote(num_frames=num_frames, num_angles=num_angles)
        print(f"\nResult: {result}")

    elif action == "verify":
        print("Verifying extracted data...")
        result = verify_extracted_data.remote()
        print(f"\nResult: {result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: extract, verify")
