"""
GUAVA Inference on Modal
========================
Based on modal_app.py lessons learned:
1. NEVER use debian_slim - always use nvidia/cuda:11.8.0-devel
2. Build PyTorch3D from source with FORCE_CUDA=1
3. Pin numpy==1.26.4 LAST
4. Use --index-url for PyTorch packages
"""

import modal
import os
import subprocess

# Volume definitions
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
guava_volume = modal.Volume.from_name("guava-results", create_if_missing=True)

# Base image with CUDA 11.8 development environment (per modal_app.py)
cuda_base = modal.Image.from_registry(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    add_python="3.10"
)

image = (
    cuda_base
    # System dependencies (modal_app.py pattern)
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
        "ffmpeg",
        "libusb-1.0-0",
        "libasound2",
    )
    # PyTorch with CUDA 11.8 (modal_app.py pattern: index_url parameter)
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    # Upgrade pip/setuptools/wheel before chumpy (--no-build-isolation needs wheel in env)
    .run_commands("pip install --upgrade pip setuptools wheel")
    # chumpy needs --no-build-isolation (setup.py does 'import pip')
    .run_commands("pip install chumpy --no-build-isolation")
    # Core dependencies (modal_app.py pattern)
    .pip_install(
        "lightning==2.2.0",
        "roma==1.5.3",
        "imageio[pyav]",
        "imageio[ffmpeg]",
        "lmdb==1.6.2",
        "open3d==0.19.0",
        "plyfile==1.0.3",
        "omegaconf==2.3.0",
        "colored==2.3.0",
        "rich==14.0.0",
        "opencv-python==4.11.0.86",
        "xformers==0.0.24",
        "gradio",
        "tyro==0.8.0",
        "onnxruntime-gpu==1.18",
        "onnx==1.16",
        "mediapipe==0.10.21",
        "easydict==1.13",
        "kornia",
        "transformers==4.57.0",
        "configer==1.3.1",
        "torchgeometry==0.1.2",
        "pynvml==13.0.1",
        "scipy",
        "tqdm",
        "einops",
        "fvcore",
        "iopath",
        "smplx[all]",
        "pyrender",
        "trimesh",
    )
    # PyTorch3D from source with CUDA (--no-build-isolation to access torch)
    .run_commands(
        "pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'",
        env={
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
        },
    )
    # Submodules build (per README, --no-build-isolation to access torch)
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && pip install --no-build-isolation .",
        "cd /root/GUAVA/submodules/simple-knn && pip install --no-build-isolation .",
        "cd /root/GUAVA/submodules/fused-ssim && pip install --no-build-isolation .",
        env={
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        },
    )
    # Pin numpy LAST to prevent overwrites (modal_app.py pattern)
    .pip_install("numpy==1.26.4")
    # Project assets
    .add_local_dir("./assets", remote_path="/root/GUAVA/assets")
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
    .add_local_dir("./dataset", remote_path="/root/GUAVA/dataset")
    .add_local_dir("./utils", remote_path="/root/GUAVA/utils")
    .add_local_dir("./configs", remote_path="/root/GUAVA/configs")
)

app = modal.App("guava-inference")

@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={"/root/EHM_results": ehm_volume, "/root/GUAVA/outputs": guava_volume}
)
def run_guava():
    os.chdir("/root/GUAVA")

    # Physical file check
    required_files = ["main/test.py"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n--- [ERROR] Missing files: {missing} ---")
        return

    # Environment
    env = os.environ.copy()
    env["MEDIAPIPE_DISABLE_GPU"] = "1"
    env["PYTHONPATH"] = "/root/GUAVA"

    # Inference command (per README)
    cmd = [
        "python", "main/test.py",
        "-d", "0",
        "-m", "assets/GUAVA",
        "-s", "outputs/driving_avatar",
        "--data_path", "/root/EHM_results/processed_data/driving"
    ]

    print("--- Starting GUAVA inference ---")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    guava_volume.commit()
    print("--- [SUCCESS] Inference completed ---")

@app.local_entrypoint()
def main():
    run_guava.remote()
