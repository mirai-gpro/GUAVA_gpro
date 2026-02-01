"""
Modal Debug Script - シンプルなイメージビルドテスト
"""

import modal

# 最小限のイメージ (参考ファイルと同じdebian_slim)
simple_image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "numpy==1.23.5",
        "chumpy",
    )
)

app = modal.App("guava-debug")


@app.function(image=simple_image, timeout=300)
def test_chumpy():
    """chumpy のインポートテスト"""
    import numpy as np
    print(f"numpy version: {np.__version__}")

    import chumpy
    print("chumpy imported successfully!")

    return {"success": True, "numpy_version": np.__version__}


@app.local_entrypoint()
def main():
    print("Testing simple image with chumpy...")
    result = test_chumpy.remote()
    print(f"Result: {result}")
