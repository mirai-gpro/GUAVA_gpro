import onnxruntime as ort
import numpy as np
import os

def check_model():
    model_path = "template_decoder.onnx" # カレントディレクトリでチェック
    
    print(f"🔍 Checking model: {model_path}")
    if not os.path.exists(model_path):
        print("❌ ファイルが見つかりません。")
        return

    # ファイルサイズ確認
    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"📦 File Size: {size_mb:.2f} MB")
    
    if size_mb < 0.1:
        print("❌ ファイルが小さすぎます！重みが入っていません。")
        return

    try:
        # ONNX Runtimeでロード
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully by ONNX Runtime.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    # ダミー入力作成
    # TypeScript側と同じ条件: N=100
    num_vertices = 100
    
    # fused_features: [1, N, 512]
    # stats from logs: min=-5, max=5 (approx)
    fused = np.random.uniform(-5, 5, (1, num_vertices, 512)).astype(np.float32)
    
    # view_dirs: [1, 27]
    dirs = np.zeros((1, 27), dtype=np.float32)
    dirs[0, 2] = 1.0 # Z-axis

    # 推論実行
    print("🏃 Running inference check...")
    try:
        inputs = {
            'fused_features': fused,
            'view_dirs': dirs
        }
        outputs = session.run(None, inputs)
        
        # 出力チェック
        output_names = [o.name for o in session.get_outputs()]
        has_nan = False
        
        for name, val in zip(output_names, outputs):
            if np.isnan(val).any() or np.isinf(val).any():
                print(f"❌ Output '{name}' contains NaN or Inf!")
                has_nan = True
                # 統計を表示
                print(f"   Min: {np.min(val)}, Max: {np.max(val)}")
            else:
                print(f"✅ Output '{name}': OK (Mean: {np.mean(val):.4f})")
        
        if not has_nan:
            print("\n🎉 MODEL IS HEALTHY!")
            print("Python上では正常に動作しています。")
            print("→ ブラウザのキャッシュが原因の可能性が高いです。")
            print("→ ブラウザで『スーパーリロード (Ctrl+F5 / Cmd+Shift+R)』を試してください。")
        else:
            print("\n💀 MODEL IS BROKEN.")
            print("Python上でもNaNが出ます。重みのロード手順に問題があります。")

    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    check_model()