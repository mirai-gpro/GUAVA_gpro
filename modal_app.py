"""
GUAVA Modal App - Environment Setup and Testing
================================================

This module defines the Modal image for GUAVA with proper CUDA support.
Use this for environment validation before running full inference.

Lessons learned from past failures:
1. NEVER use debian_slim - always use nvidia/cuda:11.8.0-devel
2. Build PyTorch3D from source with FORCE_CUDA=1
3. Pin numpy==1.26.4 LAST
4. Use --index-url for PyTorch packages
"""

import modal
import os

# ============================================================================
# Image Definition
# ============================================================================

# Base image with CUDA 11.8 development environment
cuda_base = modal.Image.from_registry(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    add_python="3.10"
)

# Full GUAVA image with all dependencies
guava_image = (
    cuda_base
    # System dependencies
    .apt_install(
        "build-essential",
        "ninja-build",
        "git",
        "cmake",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "wget",
        "curl",
    )
    # PyTorch with CUDA 11.8
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    # Core dependencies
    .pip_install(
        "lightning==2.2.0",
        "roma==1.5.3",
        "imageio[pyav]",
        "lmdb==1.6.2",
        "plyfile==1.0.3",
        "omegaconf==2.3.0",
        "opencv-python==4.11.0.86",
        "chumpy==0.70",
        "scipy",
        "tqdm",
        "einops",
        "fvcore",
        "iopath",
        "rich",
    )
    # PyTorch3D from source with CUDA
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'",
        env={
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
        },
    )
    # Pin numpy LAST to prevent overwrites
    .pip_install("numpy==1.26.4")
)

# ============================================================================
# Modal App
# ============================================================================

app = modal.App("guava-factory")

# Volumes
weights_volume = modal.Volume.from_name("guava-weights", create_if_missing=True)
ehm_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=False)


@app.function(
    image=guava_image,
    gpu="L4",
    timeout=600,
)
def check_environment():
    """
    Validate the GUAVA environment.
    Run this first to ensure all dependencies are correctly installed.
    """
    import torch
    import numpy as np

    results = {
        "status": "checking",
        "checks": {},
    }

    # Check CUDA
    cuda_ok = torch.cuda.is_available()
    results["checks"]["cuda"] = {
        "ok": cuda_ok,
        "version": torch.version.cuda if cuda_ok else None,
        "device": torch.cuda.get_device_name(0) if cuda_ok else None,
    }

    # Check PyTorch
    results["checks"]["pytorch"] = {
        "ok": True,
        "version": torch.__version__,
    }

    # Check numpy version (must be < 2.0)
    np_version = np.__version__
    np_ok = int(np_version.split(".")[0]) < 2
    results["checks"]["numpy"] = {
        "ok": np_ok,
        "version": np_version,
    }

    # Check PyTorch3D
    try:
        import pytorch3d
        results["checks"]["pytorch3d"] = {
            "ok": True,
            "version": getattr(pytorch3d, "__version__", "unknown"),
        }
    except ImportError as e:
        results["checks"]["pytorch3d"] = {
            "ok": False,
            "error": str(e),
        }

    # Check diff-gaussian-rasterization
    try:
        from diff_gaussian_rasterization_32 import GaussianRasterizer_32
        results["checks"]["diff_gaussian_rasterization"] = {"ok": True}
    except ImportError as e:
        results["checks"]["diff_gaussian_rasterization"] = {
            "ok": False,
            "error": str(e),
        }

    # Overall status
    all_ok = all(
        check.get("ok", False)
        for check in results["checks"].values()
    )
    results["status"] = "ready" if all_ok else "incomplete"

    return results


@app.function(
    image=guava_image,
    gpu="L4",
    volumes={
        "/weights": weights_volume,
        "/ehm": ehm_volume,
    },
    mounts=[modal.Mount.from_local_dir(".", remote_path="/app/guava")],
    timeout=1800,
)
def run_inference_test():
    """
    Test inference with mounted volumes.
    """
    import sys
    sys.path.insert(0, "/app/guava")

    # List available data
    print("=== Checking volumes ===")

    print("\n/weights contents:")
    for root, dirs, files in os.walk("/weights"):
        level = root.replace("/weights", "").count(os.sep)
        if level > 2:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:5]:
            print(f"{indent}  {f}")

    print("\n/ehm contents:")
    for root, dirs, files in os.walk("/ehm"):
        level = root.replace("/ehm", "").count(os.sep)
        if level > 3:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:3]:
            print(f"{indent}  {f}")

    print("\n/app/guava contents:")
    for item in os.listdir("/app/guava"):
        print(f"  {item}")

    return {"status": "volume_check_complete"}


@app.local_entrypoint()
def main():
    """Run environment check."""
    print("Checking GUAVA environment...")
    result = check_environment.remote()

    print("\n=== Environment Check Results ===")
    print(f"Status: {result['status']}")
    print("\nComponent checks:")
    for name, check in result["checks"].items():
        status = "OK" if check.get("ok") else "FAILED"
        version = check.get("version", "")
        error = check.get("error", "")
        print(f"  {name}: {status} {version} {error}")

    if result["status"] == "ready":
        print("\nEnvironment is ready for GUAVA inference!")
    else:
        print("\nSome components are missing. Please check the errors above.")
