import modal
import os
import shutil
import subprocess

# 結果保存用（名前を固定）
output_volume = modal.Volume.from_name("ehm-tracker-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", 
        "libsm6", "libxext6", "libxrender-dev", "libusb-1.0-0", "libasound2", "wget"
    )
    .pip_install(
        "torch", "torchvision", "torchaudio", "numpy==1.23.5", "chumpy", "opencv-python-headless",
        "protobuf==3.20.3", "mediapipe==0.10.11", "smplx[all]", "pyrender", "trimesh", "ninja",
        "pyyaml", "scipy", "tqdm", "tyro", "rich", "loguru", "imageio", "imageio-ffmpeg",
        "fvcore", "iopath", "lmdb", "onnxruntime-gpu", "roma", "filterpy", "scikit-image",
        "einops", "easydict", "matplotlib", "scikit-learn", "kornia", "yacs", "av",
        "transformers", "accelerate"
    )
    .pip_install("git+https://github.com/facebookresearch/pytorch3d.git")
    .add_local_dir("./assets", remote_path="/root/EHM-Tracker/assets")
    .add_local_dir("./pretrained", remote_path="/root/EHM-Tracker/pretrained")
    .add_local_dir("./data", remote_path="/root/EHM-Tracker/data")
    .add_local_dir("./src", remote_path="/root/EHM-Tracker/src")
    .add_local_file("./tracking_video.py", remote_path="/root/EHM-Tracker/tracking_video.py")
)

app = modal.App("ehm-tracker-final")

@app.function(image=image, gpu="a10g", timeout=3600, volumes={"/root/EHM-Tracker/output": output_volume})
def run_tracking():
    os.chdir("/root/EHM-Tracker")
    
    # 拡張子補正（pth -> pt）
    m_dir = "./pretrained/matting"
    if os.path.exists(f"{m_dir}/stylematte_synth.pth"):
        shutil.copy(f"{m_dir}/stylematte_synth.pth", f"{m_dir}/stylematte_synth.pt")

    # クラウド側での全ファイル物理チェック（これが通れば絶対に走り抜けます）
    required_files = [
        "./assets/SMPLX/SMPLX_NEUTRAL_2020.npz",
        "./pretrained/pixie/pixie_model.tar",
        "./pretrained/lmk70/landmark.onnx",
        "./pretrained/mediapipe/face_landmarker.task",
        "./pretrained/teaser/TEASER_lmk_2.onnx",
        "./pretrained/hamer/hamer.onnx"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("\n--- [致命的エラー] PCからの転送に失敗しています ---")
        for m in missing:
            print(f"不足: {m}")
        return

    # 実働設定（MediaPipeのクラッシュを防止）
    env = os.environ.copy()
    env["MEDIAPIPE_DISABLE_GPU"] = "1"
    
    cmd = [
        "python", "tracking_video.py", 
        "--in-root", "./data/videos", 
        "--output-dir", "./output/processed_data",
        "--v", "0" 
    ]
    
    print("--- 2026年最新リポジトリ準拠の構成で解析を開始します ---")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    output_volume.commit()
    print("--- [成功] 解析がすべて終了しました！ ---")