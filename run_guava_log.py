import os, subprocess, modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "git", "ninja-build", 
        "build-essential", "libglm-dev", "clang", "dos2unix", "ffmpeg" # ffmpegを追加
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda", 
        "MAX_JOBS": "4", "CC": "clang", "CXX": "clang++", "TORCH_CUDA_ARCH_LIST": "8.9"
    })
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install \"numpy==1.26.4\" \"scipy\""
    )
    .run_commands("pip install chumpy --no-build-isolation")
    .run_commands("pip install \"torch==2.1.0\" \"torchvision==0.16.0\" --extra-index-url https://download.pytorch.org/whl/cu121")
    .pip_install(
        "lightning", "pytorch-lightning", "omegaconf", "gsplat",
        "opencv-python", "h5py", "tqdm", "scikit-image", "trimesh", "plyfile",
        "lmdb", "lpips", "open3d", "roma", "smplx", "yacs", "ninja",
        "colored", "termcolor", "tabulate", "vispy", "configargparse", "portalocker",
        "fvcore", "iopath", "imageio-ffmpeg" # ここに動画書き出し用ライブラリを追加
    )
    .run_commands("pip install \"numpy==1.26.4\"")
    .run_commands("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html")
    .add_local_dir("./submodules", remote_path="/root/guava/submodules", copy=True)
    .run_commands(
        "cd /root/guava/submodules/diff-gaussian-rasterization-32 && rm -rf build && pip install . --no-build-isolation",
        "cd /root/guava/submodules/simple-knn && rm -rf build && pip install . --no-build-isolation"
    )
    .add_local_dir(".", remote_path="/root/guava", copy=True, ignore=["assets/", "submodules/", "outputs/", ".venv/"])
    .run_commands("find /root/guava -maxdepth 3 -name '*.py' -not -path '*/assets/*' | xargs dos2unix")
)

app = modal.App("guava-v1-final-clean")
volume = modal.Volume.from_name("guava-weights")

@app.function(gpu="L4", image=image, volumes={"/assets": volume}, timeout=3600)
def run_guava():
    os.chdir("/root/guava")
    
    import shutil
    if os.path.exists("assets"):
        if os.path.islink("assets"): os.unlink("assets")
        else: shutil.rmtree("assets")
    os.makedirs("assets", exist_ok=True)

    targets = ["SMPLX", "GUAVA", "FLAME", "example"]
    for target in targets:
        src = f"/assets/{target}"
        dst = f"assets/{target}"
        if os.path.exists(src): os.symlink(src, dst)

    print("\n?? [FINAL VIDEO EXPORT] Starting GUAVA Inference...")
    cmd = ["python3", "main/test.py", "--devices", "0", "--model_path", "assets/GUAVA", "--save_path", "/assets/outputs/sakura_test", "--saving_name", "render", "--data_path", "assets/example/tracked_video/6gvP8f5WQyo__056"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONPATH": "/root/guava"})
    for line in process.stdout: print(f"[GUAVA] {line}", end="")
    process.wait()
    return f"Done: {process.returncode}"

@app.local_entrypoint()
def main():
    print(run_guava.remote())