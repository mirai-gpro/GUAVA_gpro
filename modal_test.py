Set-Content -Path modal_test.py -Value @'
import modal
import os

# 1. 前回の成功したイメージ定義
guava_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.1.0", "numpy", "scipy", "gsplat",
        "opencv-python", "h5py", "tqdm"
    )
    .pip_install("git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7")
)

app = modal.App("guava-inference")

@app.function(
    gpu="L4", 
    image=guava_image,
    # Volumeのルートを /assets にマウント
    volumes={"/assets": modal.Volume.from_name("guava-weights")},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/guava")]
)
def run_guava():
    import subprocess
    os.chdir("/root/guava")
    
    # lsの結果に基づき、パスを /assets/assets/... に修正
    model_path = "/assets/assets/GUAVA"
    # ※もし example データがない場合はエラーになるので、まずはパスの存在を確認します
    if not os.path.exists(model_path):
        return f"Error: Model not found at {model_path}. Please check volume paths."

    # 推論コマンドの実行
    cmd = [
        "python", "main/test.py",
        "-d", "0",
        "-m", model_path,
        "-s", "outputs/sakura_test",
        "--data_path", "/assets/assets/example/tracked_video/6gvP8f5WQyo__056"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    return "Inference process completed! Check the logs above."

@app.local_entrypoint()
def main():
    print(run_guava.remote())
'@
