import modal
import os
import subprocess
import glob

# 正確なVolume名（履歴から確認済み）
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
guava_volume = modal.Volume.from_name("guava-results", create_if_missing=True)

image = (
    # 最先端のModal環境で、CUDA 11.8 + Python 3.10 を構築
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "libusb-1.0-0", "build-essential", "ninja-build")
    
    # 1. 基礎工事（NumPy 2.0混入を許さない固定）
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install numpy==1.26.4"
    )
    
    # 2. 王道：PyTorch3D をビルドせずバイナリで導入
    .run_commands(
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118",
        "pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/prefix/wheels/py310_cu118_pyt220/download.html"
    )
    
    # 3. 残りの全ライブラリ導入（isolationなしで環境を保護）
    .pip_install(
        "chumpy==0.70", "lightning==2.2.0", "roma==1.5.3", "imageio[pyav]", "imageio[ffmpeg]",
        "lmdb==1.6.2", "open3d==0.19.0", "plyfile==1.0.3", "omegaconf==2.3.0",
        "rich==14.0.0", "opencv-python-headless", "xformers==0.0.24",
        "tyro==0.8.0", "onnxruntime-gpu==1.18", "onnx==1.16", "mediapipe==0.10.21",
        "transformers==4.37.0", "configer==1.3.1", "torchgeometry==0.1.2", "pynvml==13.0.1",
        # NumPyを最後に再度念押しで固定
        "numpy==1.26.4"
    )
    
    # 4. サブモジュールのビルド（隔離なし）
    .env({"FORCE_CUDA": "1", "CUDA_HOME": "/usr/local/cuda", "MEDIAPIPE_DISABLE_GPU": "1"})
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules")
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/fused-ssim && pip install . --no-build-isolation"
    )
    
    # 5. アセットおよびメインコードの転送
    .add_local_dir("./assets", remote_path="/root/GUAVA/assets")
    .add_local_dir("./main", remote_path="/root/GUAVA/main")
    .add_local_dir("./models", remote_path="/root/GUAVA/models")
)

app = modal.App("guava-gpro-final")

@app.function(
    image=image, 
    gpu="a10g", 
    timeout=3600, 
    volumes={
        "/root/EHM_results": ehm_volume,    
        "/root/GUAVA/outputs": guava_volume 
    }
)
def run_guava():
    os.chdir("/root/GUAVA")
    
    # --- 地雷除去1: 昨日のボリューム内パスの自動探索 ---
    # processed_data/driving か processed_data/driving/driving かを自動判別
    search_path = "/root/EHM_results/processed_data/driving"
    if os.path.exists(os.path.join(search_path, "driving")):
        target_data_path = os.path.join(search_path, "driving")
    else:
        target_data_path = search_path
    
    print(f"--- DETECTED DATA PATH: {target_data_path} ---")

    # --- 地雷除去2: YAMLファイル内のWindows絶対パスをコンテナパスに全置換 ---
    for yaml_file in glob.glob("assets/GUAVA/*.yaml"):
        with open(yaml_file, 'r') as f:
            content = f.read()
        if "C:\\" in content or "C:/" in content:
            print(f"--- Cleaning Windows paths in {yaml_file} ---")
            # Windows形式のパスをコンテナ内の assets パスへ強引に書き換え
            import re
            content = re.sub(r'[A-Z]:\\.*\\assets', '/root/GUAVA/assets', content)
            with open(yaml_file, 'w') as f:
                f.write(content)

    # --- 推論実行 ---
    cmd = [
        "python", "main/test.py",
        "-d", "0",
        "-m", "assets/GUAVA",
        "-s", "outputs/driving_avatar",
        "--data_path", target_data_path
    ]
    
    print("--- STARTING GUAVA AVATAR GENERATION ---")
    subprocess.run(cmd, check=True)
    
    guava_volume.commit()
    print("--- SUCCESS: Check results in guava-results volume ---")

@app.local_entrypoint()
def main():
    run_guava.remote()