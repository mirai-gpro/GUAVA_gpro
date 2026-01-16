import modal
import os
import subprocess
import shutil
import glob

# --- 1. Volume設定（全てクラウド上の資産を使う） ---
# (1) 動きのデータ (run_ehm.pyの成果物)
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
# (2) 見た目のモデル (女性コンシェルジュが入っているはずの場所)
weights_volume = modal.Volume.from_name("guava-weights")
# (3) 結果の保存先
guava_volume = modal.Volume.from_name("guava-results", create_if_missing=True)

# --- 2. 環境構成（完走実績 run_guava_log.py を100%維持） ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "git", "ninja-build", 
        "build-essential", "libglm-dev", "clang", "dos2unix", "ffmpeg", 
        "libsm6", "libxext6", "libxrender-dev"
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
        "fvcore", "iopath", "imageio-ffmpeg"
    )
    .run_commands("pip install \"numpy==1.26.4\"")
    .run_commands("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html")
    
    # --- 3. ファイル配置 ---
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && rm -rf build && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && rm -rf build && pip install . --no-build-isolation"
    )
    # PCからはコードだけを送る（assetsは送らない）
    .add_local_dir(".", remote_path="/root/GUAVA", copy=True, ignore=["assets/", "outputs/", ".venv/", ".git/"])
    .run_commands("find /root/GUAVA -maxdepth 3 -name '*.py' | xargs dos2unix")
)

app = modal.App("guava-gpro-cloud-assets")

@app.function(
    gpu="L4", 
    image=image, 
    # ここで3つのVolumeをすべてマウントします
    volumes={
        "/root/EHM_results": ehm_volume,    # 動きデータ
        "/root/GUAVA/assets": weights_volume, # 見た目モデル (ここに女性がいるはず)
        "/root/GUAVA/outputs": guava_volume   # 保存先
    },
    timeout=3600
)
def run_guava():
    os.chdir("/root/GUAVA")
    
    # --- 物理チェック ---
    required_files = ["main/test.py"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n--- [ERROR] コードが見つかりません: {missing} ---")
        return

    # Assetsの確認（クラウド上のVolumeの中身を見る）
    print("--- Checking Assets in Cloud Volume ---")
    if os.path.exists("assets"):
        print(f"Assets found: {os.listdir('assets')}")
    else:
        print("--- [WARNING] Assets folder is empty! ---")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/GUAVA"
    
    # EHMパス探索
    search_path = "/root/EHM_results/processed_data/driving"
    target_data_path = ""
    
    if os.path.exists(search_path):
        potential_path = os.path.join(search_path, "driving")
        if os.path.exists(potential_path):
             target_data_path = potential_path
        else:
             target_data_path = search_path
    else:
        print(f"--- [WARNING] EHM Data path not found at {search_path}. Checking root... ---")
        if os.path.exists("/root/EHM_results"):
            target_data_path = "/root/EHM_results"
        else:
            print("--- [FATAL] Data not found anywhere. ---")

    print(f"--- Target Data Path: {target_data_path} ---")

    # 推論コマンド
    # -m assets/GUAVA は、PCのフォルダではなく、マウントされた Volume (guava-weights) の中身を参照します
    cmd = [
        "python", "main/test.py",
        "-d", "0",
        "-m", "assets/GUAVA", 
        "-s", "outputs/driving_avatar",
        "--data_path", target_data_path
    ]

    print("--- Starting GUAVA Inference (Using Cloud Assets & Data) ---")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    if process.returncode == 0:
        guava_volume.commit()
        print("--- [SUCCESS] Inference completed & Committed to Volume ---")
    else:
        print(f"--- [FAILED] Inference failed with exit code {process.returncode} ---")
        raise RuntimeError("Inference failed")

@app.local_entrypoint()
def main():
    run_guava.remote()