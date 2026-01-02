"""
GUAVA 3D Gaussian Splatting Avatar - Modal Deployment
======================================================

This script deploys GUAVA on Modal with a properly configured CUDA environment.

Key design decisions to avoid past failures:
1. Use nvidia/cuda:11.8.0-devel-ubuntu22.04 as base (not debian_slim)
2. Build PyTorch3D from source with FORCE_CUDA=1 (no binary URLs)
3. Pin numpy<2.0 at the END of all installations
4. Auto-detect data paths instead of hardcoding
5. Mount EHM-Tracker output volume correctly

Usage:
    modal run run_guava_gpro.py --data-subpath processed_data/driving/YOUR_VIDEO_NAME
"""

import modal
import os
from pathlib import Path

# ============================================================================
# Image Definition - Built from CUDA devel base
# ============================================================================

def create_guava_image():
    """
    Create a Modal image with all GUAVA dependencies properly installed.

    Build order is critical:
    1. System packages (build-essential, ninja-build)
    2. PyTorch with CUDA 11.8
    3. PyTorch3D from source (requires CUDA headers)
    4. GUAVA submodules (diff-gaussian-rasterization-32, etc.)
    5. Other Python dependencies
    6. numpy<2.0 pinned LAST
    """

    # Start from NVIDIA CUDA 11.8 devel image (has nvcc and headers)
    image = modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10"
    )

    # Install system dependencies
    image = image.apt_install(
        "build-essential",
        "ninja-build",
        "git",
        "cmake",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "curl",
    )

    # Install PyTorch 2.2.0 with CUDA 11.8
    image = image.pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )

    # Install core dependencies before PyTorch3D
    image = image.pip_install(
        "lightning==2.2.0",
        "roma==1.5.3",
        "imageio[pyav]",
        "imageio[ffmpeg]",
        "lmdb==1.6.2",
        "plyfile==1.0.3",
        "omegaconf==2.3.0",
        "colored==2.3.0",
        "rich==14.0.0",
        "opencv-python==4.11.0.86",
        "chumpy==0.70",
        "tyro==0.8.0",
        "easydict==1.13",
        "kornia==0.7.0",  # Updated from 0.1.9 for compatibility
        "transformers==4.40.0",  # Pinned for stability
        "configer==1.3.1",
        "torchgeometry==0.1.2",
        "pynvml==13.0.1",
        "scipy",
        "tqdm",
        "einops",
        "fvcore",
        "iopath",
    )

    # Build PyTorch3D from source with CUDA support
    image = image.run_commands(
        "pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'",
        env={
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
        },
    )

    # Pin numpy<2.0 to prevent compatibility issues
    # This MUST be done after all other installations
    image = image.pip_install("numpy==1.26.4")

    # Install xformers for attention optimization
    image = image.pip_install(
        "xformers==0.0.24",
        index_url="https://download.pytorch.org/whl/cu118",
    )

    return image


# Create the image
guava_image = create_guava_image()

# ============================================================================
# Modal App and Volume Configuration
# ============================================================================

app = modal.App("guava-3dgs-avatar")

# Volume for EHM-Tracker output (Phase 1 results)
ehm_output_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=False)

# Volume for GUAVA output (rendered results)
guava_output_volume = modal.Volume.from_name("guava-output", create_if_missing=True)

# Volume for model weights
guava_weights_volume = modal.Volume.from_name("guava-weights", create_if_missing=True)


# ============================================================================
# Helper Functions
# ============================================================================

def find_tracked_data(base_path: str, pattern: str = "processed_data") -> list:
    """
    Recursively find all directories containing tracked data.
    Returns list of paths that contain 'optim_tracking_ehm.pkl' or similar markers.
    """
    results = []
    base = Path(base_path)

    if not base.exists():
        return results

    # Look for tracking result markers
    markers = ["optim_tracking_ehm.pkl", "tracking.pkl", "smplx_params.pkl"]

    for root, dirs, files in os.walk(base_path):
        for marker in markers:
            if marker in files:
                results.append(root)
                break

    return results


def list_volume_contents(volume_path: str, max_depth: int = 3) -> str:
    """List contents of a volume path for debugging."""
    output = []
    base = Path(volume_path)

    if not base.exists():
        return f"Path does not exist: {volume_path}"

    for root, dirs, files in os.walk(volume_path):
        depth = root.replace(volume_path, "").count(os.sep)
        if depth > max_depth:
            continue

        indent = "  " * depth
        output.append(f"{indent}{os.path.basename(root)}/")

        subindent = "  " * (depth + 1)
        for file in files[:5]:  # Limit files shown per directory
            output.append(f"{subindent}{file}")
        if len(files) > 5:
            output.append(f"{subindent}... and {len(files) - 5} more files")

    return "\n".join(output[:50])  # Limit total output


# ============================================================================
# Image with pre-built submodules
# ============================================================================

# Create image with submodules built in
# Modal 1.0+: use add_local_dir(copy=True) instead of copy_local_dir
guava_full_image = (
    guava_image
    .add_local_dir("submodules/diff-gaussian-rasterization-32", "/opt/guava/submodules/diff-gaussian-rasterization-32", copy=True)
    .add_local_dir("submodules/simple-knn", "/opt/guava/submodules/simple-knn", copy=True)
    .add_local_dir("submodules/fused-ssim", "/opt/guava/submodules/fused-ssim", copy=True)
    .add_local_dir("submodules/lpipsPyTorch", "/opt/guava/submodules/lpipsPyTorch", copy=True)
    .run_commands(
        "cd /opt/guava/submodules/diff-gaussian-rasterization-32 && pip install -e .",
        env={"FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0"},
    )
    .run_commands(
        "cd /opt/guava/submodules/simple-knn && pip install -e .",
        env={"FORCE_CUDA": "1"},
    )
    .run_commands(
        "cd /opt/guava/submodules/fused-ssim && pip install -e .",
        env={"FORCE_CUDA": "1"},
    )
    # Final numpy pin after all builds
    .pip_install("numpy==1.26.4")
    # Add GUAVA source code (Modal 1.0+ uses add_local_dir instead of Mount)
    .add_local_dir(".", "/app/guava", copy=False)
)


# ============================================================================
# Main Inference Function
# ============================================================================

@app.function(
    image=guava_full_image,
    gpu="L4",
    timeout=3600,  # 1 hour
    volumes={
        "/ehm_output": ehm_output_volume,
        "/guava_output": guava_output_volume,
        "/weights": guava_weights_volume,
    },
)
def run_guava_inference(
    data_subpath: str = None,
    source_subpath: str = None,
    skip_self_act: bool = False,
    render_cross_act: bool = False,
):
    """
    Run GUAVA inference on tracked data from EHM-Tracker.

    Args:
        data_subpath: Path within ehm_output volume to the tracked data
                     (e.g., "processed_data/driving/my_video")
        source_subpath: Optional source data path for cross-reenactment
        skip_self_act: Skip self-reenactment rendering
        render_cross_act: Enable cross-reenactment rendering

    Returns:
        dict with status and output paths
    """
    import subprocess
    import sys
    import shutil

    os.chdir("/app/guava")
    sys.path.insert(0, "/app/guava")

    results = {
        "status": "starting",
        "steps": [],
    }

    # Step 1: Verify CUDA and PyTorch
    print("=" * 60)
    print("Step 1: Verifying CUDA and PyTorch")
    print("=" * 60)

    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    results["cuda"] = {
        "available": cuda_available,
        "device": cuda_device,
        "torch_version": torch.__version__,
    }
    print(f"CUDA Available: {cuda_available}")
    print(f"CUDA Device: {cuda_device}")
    print(f"PyTorch Version: {torch.__version__}")

    if not cuda_available:
        results["status"] = "error"
        results["error"] = "CUDA not available"
        return results

    results["steps"].append("cuda_verified")

    # Step 2: Verify submodule installations
    print("\n" + "=" * 60)
    print("Step 2: Verifying submodule installations")
    print("=" * 60)

    try:
        from diff_gaussian_rasterization_32 import GaussianRasterizer_32
        print("diff-gaussian-rasterization-32: OK")
        results["steps"].append("dgr32_verified")
    except ImportError as e:
        print(f"diff-gaussian-rasterization-32: FAILED - {e}")
        results["status"] = "error"
        results["error"] = f"diff-gaussian-rasterization-32 import failed: {e}"
        return results

    # Step 3: Verify EHM output volume
    print("\n" + "=" * 60)
    print("Step 3: Checking EHM-Tracker output volume")
    print("=" * 60)

    ehm_path = Path("/ehm_output")
    if not ehm_path.exists():
        results["status"] = "error"
        results["error"] = "EHM output volume not mounted"
        return results

    print("EHM Output Volume Contents:")
    print(list_volume_contents("/ehm_output", max_depth=3))

    # Find tracked data automatically if not specified
    if data_subpath is None:
        print("\nAuto-detecting tracked data...")
        tracked_dirs = find_tracked_data("/ehm_output")
        if tracked_dirs:
            data_subpath = tracked_dirs[0].replace("/ehm_output/", "")
            print(f"Auto-detected: {data_subpath}")
        else:
            results["status"] = "error"
            results["error"] = "No tracked data found in EHM output volume"
            return results

    full_data_path = f"/ehm_output/{data_subpath}"
    if not Path(full_data_path).exists():
        results["status"] = "error"
        results["error"] = f"Data path does not exist: {full_data_path}"
        return results

    print(f"Using data path: {full_data_path}")
    results["data_path"] = full_data_path
    results["steps"].append("data_path_verified")

    # Step 4: Check model weights
    print("\n" + "=" * 60)
    print("Step 4: Checking model weights")
    print("=" * 60)

    # Check local assets first, then weights volume
    model_paths_to_check = [
        "/app/guava/assets/GUAVA",
        "/weights/GUAVA",
        "/weights/assets/GUAVA",
    ]

    model_path = None
    for mp in model_paths_to_check:
        if Path(mp).exists() and Path(f"{mp}/config.yaml").exists():
            model_path = mp
            print(f"Found model at: {model_path}")
            break

    if model_path is None:
        print("Model weights not found. Checking weights volume contents:")
        print(list_volume_contents("/weights", max_depth=2))
        results["status"] = "error"
        results["error"] = "GUAVA model weights not found. Please download and place in weights volume."
        return results

    results["model_path"] = model_path
    results["steps"].append("model_verified")

    # Step 5: Check parametric models (SMPLX, FLAME)
    print("\n" + "=" * 60)
    print("Step 5: Checking parametric models")
    print("=" * 60)

    smplx_path = Path("/app/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz")
    flame_path = Path("/app/guava/assets/FLAME/FLAME2020/generic_model.pkl")

    missing_models = []
    if not smplx_path.exists():
        # Check weights volume
        alt_smplx = Path("/weights/SMPLX/SMPLX_NEUTRAL_2020.npz")
        if alt_smplx.exists():
            smplx_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(alt_smplx, smplx_path)
            print("SMPLX model copied from weights volume")
        else:
            missing_models.append("SMPLX_NEUTRAL_2020.npz")
    else:
        print("SMPLX model: OK")

    if not flame_path.exists():
        alt_flame = Path("/weights/FLAME/generic_model.pkl")
        if alt_flame.exists():
            flame_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(alt_flame, flame_path)
            # Also copy to SMPLX folder as required
            smplx_flame = Path("/app/guava/assets/SMPLX/flame_generic_model.pkl")
            shutil.copy(alt_flame, smplx_flame)
            print("FLAME model copied from weights volume")
        else:
            missing_models.append("generic_model.pkl (FLAME)")
    else:
        print("FLAME model: OK")

    if missing_models:
        results["status"] = "error"
        results["error"] = f"Missing parametric models: {missing_models}"
        return results

    results["steps"].append("parametric_models_verified")

    # Step 6: Run inference
    print("\n" + "=" * 60)
    print("Step 6: Running GUAVA inference")
    print("=" * 60)

    output_path = "/guava_output/renders"
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        sys.executable, "main/test.py",
        "-d", "0",
        "-m", model_path,
        "-s", output_path,
        "--data_path", full_data_path,
    ]

    if skip_self_act:
        cmd.append("--skip_self_act")

    if render_cross_act and source_subpath:
        source_full_path = f"/ehm_output/{source_subpath}"
        if Path(source_full_path).exists():
            cmd.extend([
                "--source_data_path", source_full_path,
                "--render_cross_act"
            ])

    print(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "/app/guava"}
    )

    print("\n--- STDOUT ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    if result.returncode != 0:
        results["status"] = "error"
        results["error"] = f"Inference failed with code {result.returncode}"
        results["stderr"] = result.stderr[-2000:] if result.stderr else ""
        return results

    results["status"] = "success"
    results["output_path"] = output_path
    results["steps"].append("inference_complete")

    # Commit output volume
    guava_output_volume.commit()

    return results


# ============================================================================
# Debug Functions
# ============================================================================

@app.function(
    image=guava_full_image,
    gpu="L4",
    volumes={
        "/ehm_output": ehm_output_volume,
        "/weights": guava_weights_volume,
    },
)
def debug_check_environment():
    """Check the environment and list available data."""
    import torch
    import numpy as np

    info = {
        "python_version": os.popen("python --version").read().strip(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    # Check GPU
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

    # Check volumes
    info["ehm_output_contents"] = list_volume_contents("/ehm_output", max_depth=3)
    info["weights_contents"] = list_volume_contents("/weights", max_depth=2)

    # Check submodules
    try:
        import diff_gaussian_rasterization_32
        info["dgr32"] = "OK"
    except ImportError as e:
        info["dgr32"] = f"FAILED: {e}"

    try:
        import pytorch3d
        info["pytorch3d"] = f"OK - {pytorch3d.__version__}"
    except ImportError as e:
        info["pytorch3d"] = f"FAILED: {e}"

    return info


@app.function(
    image=guava_full_image,
    volumes={
        "/ehm_output": ehm_output_volume,
    },
)
def list_ehm_data():
    """List all tracked data in the EHM output volume."""
    tracked = find_tracked_data("/ehm_output")
    return {
        "tracked_data_paths": tracked,
        "volume_tree": list_volume_contents("/ehm_output", max_depth=4),
    }


# ============================================================================
# Entry Points
# ============================================================================

@app.local_entrypoint()
def main(
    data_subpath: str = None,
    source_subpath: str = None,
    debug: bool = False,
    list_data: bool = False,
    cross_act: bool = False,
):
    """
    Main entry point for GUAVA inference.

    Examples:
        # Debug environment
        modal run modal_run_guava.py --debug

        # List available tracked data
        modal run modal_run_guava.py --list-data

        # Run inference with auto-detected data
        modal run modal_run_guava.py

        # Run inference with specific data path
        modal run modal_run_guava.py --data-subpath processed_data/driving/my_video

        # Cross-reenactment
        modal run modal_run_guava.py --data-subpath target_video --source-subpath source_image --cross-act
    """

    if debug:
        print("Running environment check...")
        result = debug_check_environment.remote()
        print("\n=== Environment Info ===")
        for key, value in result.items():
            if isinstance(value, str) and "\n" in value:
                print(f"\n{key}:")
                print(value)
            else:
                print(f"{key}: {value}")
        return

    if list_data:
        print("Listing tracked data in EHM output volume...")
        result = list_ehm_data.remote()
        print("\n=== Tracked Data Paths ===")
        for path in result["tracked_data_paths"]:
            print(f"  {path}")
        print("\n=== Volume Tree ===")
        print(result["volume_tree"])
        return

    print("Starting GUAVA inference...")
    result = run_guava_inference.remote(
        data_subpath=data_subpath,
        source_subpath=source_subpath,
        skip_self_act=cross_act,  # Skip self-act if doing cross-act
        render_cross_act=cross_act,
    )

    print("\n=== Inference Result ===")
    print(f"Status: {result['status']}")
    print(f"Steps completed: {result.get('steps', [])}")

    if result["status"] == "error":
        print(f"Error: {result.get('error', 'Unknown error')}")
        if "stderr" in result:
            print(f"\nStderr (last 2000 chars):\n{result['stderr']}")
    else:
        print(f"Output path: {result.get('output_path', 'N/A')}")
