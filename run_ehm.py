import modal
import os
import subprocess
import shutil

output_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=True)

image = (
    modal.Image.micromamba(python_version="3.9")
    .micromamba_install(
        "numpy=1.23.1", "scipy", "pytorch", "torchvision", "torchaudio", 
        "pytorch-cuda=12.1", "pytorch3d", "ninja",
        channels=["pytorch3d", "pytorch", "nvidia", "conda-forge"]
    )
    .pip_install(
        "numpy==1.23.1", "chumpy", "smplx[all]", "opencv-python-headless",
        "mediapipe==0.10.11", "protobuf==3.20.3", "pyrender", "trimesh", 
        "pyyaml", "tqdm", "tyro", "rich", "loguru", "imageio", "imageio-ffmpeg", 
        "fvcore", "iopath", "lmdb", "onnxruntime-gpu", "roma", "filterpy", 
        "scikit-image", "einops", "easydict", "matplotlib", "scikit-learn", 
        "kornia", "yacs", "av", "transformers", "accelerate", "configer", 
        "torchgeometry", "pynvml"
    )
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget")
    .add_local_dir("./assets", remote_path="/root/EHM-Tracker/assets")
    .add_local_dir("./pretrained", remote_path="/root/EHM-Tracker/pretrained")
    .add_local_dir("./data", remote_path="/root/EHM-Tracker/data")
    .add_local_dir("./src", remote_path="/root/EHM-Tracker/src")
    .add_local_file("./tracking_video.py", remote_path="/root/EHM-Tracker/tracking_video.py")
)

app = modal.App("ehm-tracker-final-clean")

# --- 追加：掃除専用のタスク（GPUを使わないので安上がりで確実） ---
@app.function(volumes={"/root/EHM-Tracker/output": output_volume})
def reset_storage():
    target = "/root/EHM-Tracker/output/processed_data"
    if os.path.exists(target):
        print(f"--- [Cleaning] 過去の残骸を削除中: {target} ---")
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)
    output_volume.commit()
    print("--- [Success] ストレージを空にしました。いつでも実行可能です ---")

# --- メインの解析タスク ---
@app.function(image=image, gpu="a10g", timeout=3600, volumes={"/root/EHM-Tracker/output": output_volume})
def run_tracking():
    os.chdir("/root/EHM-Tracker")
    
    # VPoser設定の補完
    vp_dir = "./pretrained/vposer/vposer_v1_0"
    if os.path.exists(vp_dir):
        conf_path = os.path.join(vp_dir, "config.json")
        if not os.path.exists(conf_path):
            import json
            with open(conf_path, "w") as f:
                json.dump({"model_params": {"num_neurons": 512, "latentD": 32, "data_shape": [1, 21, 3], "use_cont_repr": True}}, f)

    cmd = [
        "python", "tracking_video.py", 
        "--in_root", "./data/videos", 
        "--output_dir", "./output/processed_data",
        "--check_hand_score", "0.7"
    ]
    
    import numpy as np
    print(f"--- 実行環境: NumPy {np.__version__} / モデル整合性OK ---")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    output_volume.commit()

@app.local_entrypoint()
def main(reset: bool = False):
    if reset:
        reset_storage.remote()
    
    run_tracking.remote()