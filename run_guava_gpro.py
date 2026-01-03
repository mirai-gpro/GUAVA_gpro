import modal
import os
import shutil
import subprocess

# Volume definitions
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
guava_volume = modal.Volume.from_name("guava-results", create_if_missing=True)

image = (
    # CUDA devel image needed for building 3DGS submodules (per README: CUDA 11.8)
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        # modal_final_clean.py pattern
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg",
        "libsm6", "libxext6", "libxrender-dev", "libusb-1.0-0", "libasound2", "wget",
        # Additional for CUDA builds
        "build-essential", "ninja-build"
    )
    # Fix pip environment (cuda image has minimal setup, unlike debian_slim)
    .run_commands("pip install --upgrade pip setuptools wheel")
    # modal_final_clean.py pattern: simple pip_install
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "numpy==1.23.5", "chumpy", "opencv-python-headless",
        "protobuf==3.20.3", "mediapipe==0.10.11", "smplx[all]", "pyrender", "trimesh", "ninja",
        "pyyaml", "scipy", "tqdm", "tyro", "rich", "imageio", "imageio-ffmpeg",
        "fvcore", "iopath", "lmdb", "onnxruntime-gpu", "roma", "transformers"
    )
    # PyTorch3D (per README: v0.7.7)
    .env({"FORCE_CUDA": "1", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.6"})
    .pip_install("git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7")
    # GUAVA-specific packages
    .pip_install("lightning==2.2.0", "xformers", "open3d", "plyfile", "omegaconf", "configer", "torchgeometry", "pynvml")
    # Submodules build (per README)
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && pip install .",
        "cd /root/GUAVA/submodules/simple-knn && pip install .",
        "cd /root/GUAVA/submodules/fused-ssim && pip install ."
    )
    # Project assets (modal_final_clean.py pattern)
    .add_local_dir("./assets", remote_path="/root/GUAVA/assets")
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
    .add_local_dir("./dataset", remote_path="/root/GUAVA/dataset")
    .add_local_dir("./utils", remote_path="/root/GUAVA/utils")
    .add_local_dir("./configs", remote_path="/root/GUAVA/configs")
)

app = modal.App("guava-final")

@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={"/root/EHM_results": ehm_volume, "/root/GUAVA/outputs": guava_volume}
)
def run_guava():
    os.chdir("/root/GUAVA")

    # modal_final_clean.py pattern: shutil.copy for file surgery (no file writing)
    # This avoids \r corruption that occurred with open/write/re.sub

    # Physical file check (modal_final_clean.py pattern)
    required_files = ["main/test.py"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n--- [ERROR] Missing files: {missing} ---")
        return

    # Environment (modal_final_clean.py pattern)
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

    # modal_final_clean.py pattern: Popen with streaming
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    guava_volume.commit()
    print("--- [SUCCESS] Inference completed ---")

@app.local_entrypoint()
def main():
    run_guava.remote()
