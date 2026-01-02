import modal
import os
import subprocess
import glob
import re

ehm_volume = modal.Volume.from_name("ehm-tracker-output")
guava_volume = modal.Volume.from_name("guava-results", create_if_missing=True)

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

    # 2. Build environment
    .env({
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "CC": "clang",
        "CXX": "clang++"
    })

    # 3. Install requirements.txt (as README says)
    .add_local_file("./requirements.txt", remote_path="/tmp/requirements.txt", copy=True)
    .run_commands(
        "pip install chumpy==0.70 --no-build-isolation",
        "pip install -r /tmp/requirements.txt --no-build-isolation",
    )

    # 4. Install PyTorch3D (as README says)
    .run_commands(
        "pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7 --no-build-isolation"
    )

    # 5. Submodules Build (as README says: cd submodules && pip install ...)
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules && pip install diff-gaussian-rasterization-32 --no-build-isolation",
        "cd /root/GUAVA/submodules && pip install simple-knn --no-build-isolation",
        "cd /root/GUAVA/submodules && pip install fused-ssim --no-build-isolation"
    )
    
    # 5. Project Assets (最後にまとめて追加)
    .add_local_dir("./assets", remote_path="/root/GUAVA/assets")
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
    .add_local_dir("./dataset", remote_path="/root/GUAVA/dataset")
    .add_local_dir("./utils", remote_path="/root/GUAVA/utils")
    .add_local_dir("./configs", remote_path="/root/GUAVA/configs")
)

app = modal.App("guava-v3-fix-v4")

@app.function(
    image=image, 
    gpu="a10g", 
    timeout=3600, 
    volumes={"/root/EHM_results": ehm_volume, "/root/GUAVA/outputs": guava_volume},
    env={"MEDIAPIPE_DISABLE_GPU": "1"}
)
def run_guava():
    os.chdir("/root/GUAVA")

    # Add GUAVA to Python path for imports
    import sys
    sys.path.insert(0, "/root/GUAVA")

    search_path = "/root/EHM_results/processed_data/driving"
    target_data_path = os.path.join(search_path, "driving") if os.path.exists(os.path.join(search_path, "driving")) else search_path
    
    for yaml_file in glob.glob("assets/GUAVA/*.yaml"):
        with open(yaml_file, 'r') as f:
            content = f.read()
        if "C:\\" in content or "C:/" in content:
            new_content = re.sub(r'[A-Z]:[\\/].*[\\/]assets', '/root/GUAVA/assets', content)
            with open(yaml_file, 'w') as f:
                f.write(new_content)

    cmd = ["python", "main/test.py", "-d", "0", "-m", "assets/GUAVA", "-s", "outputs/driving_avatar", "--data_path", target_data_path]
    print(f"--- STARTING INFERENCE: {target_data_path} ---")

    # Set PYTHONPATH for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/GUAVA"

    # Run with output streaming to see errors
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Inference failed with exit code {process.returncode}")
    guava_volume.commit()

@app.local_entrypoint()
def main():
    run_guava.remote()