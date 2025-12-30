import modal
import os

# 1. GUAVAに必要なライブラリを網羅したイメージを定義
guava_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") # OpenCV用
    .pip_install(
        "torch",
        "numpy",
        "opencv-python",
        "gsplat",
        "scipy",
        "tqdm",
        "h5py"
    )
)

app = modal.App("guava-factory")

# 2. モデルの重み（checkpoints）を保存する領域
volume = modal.Volume.from_name("guava-weights", create_if_missing=True)

# 3. 推論関数の定義
@app.function(
    gpu="L4", 
    image=guava_image, 
    volumes={"/root/checkpoints": volume}, # 重みファイルをマウント
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/guava")] # ソースをマウント
)
def generate_sakura_3dgs():
    import sys
    sys.path.append("/root/guava")
    
    # ここでGUAVAの推論ロジックを呼び出す
    print("Starting GUAVA inference on L4 GPU...")
    # 実際には infer.py の中身をここに記述、またはインポートして実行
    return "3DGS Model (.ply) generated successfully!"

@app.local_entrypoint()
def main():
    print(generate_sakura_3dgs.remote())
