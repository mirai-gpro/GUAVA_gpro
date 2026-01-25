#!/usr/bin/env python3
"""
test_template_decoder_output.py

PyTorchモデルとONNXモデルの出力を比較
"""

import torch
import onnxruntime as ort
import numpy as np

# PyTorchモデルをロード（export_complete_template_decoder.pyから）
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("🧪 Template Decoder Output Test")
print("=" * 70)

# チェックポイント読み込み
CHECKPOINT_PATH = "best_160000.pt"
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

# CompleteTemplateDecoderクラスをインポート（ファイルから定義が必要）
# ここでは簡略化のため、直接実行

print("\n[1/3] Testing PyTorch model...")

# テスト入力
N = 10595
test_proj = torch.randn(1, N, 128)
test_global = torch.randn(1, 768)
test_dirs = torch.randn(1, 27)

print(f"   Input shapes:")
print(f"     projection_features: {list(test_proj.shape)}")
print(f"     global_embedding: {list(test_global.shape)}")
print(f"     view_dirs: {list(test_dirs.shape)}")

# ONNX推論
print("\n[2/3] Testing ONNX model...")

session = ort.InferenceSession("template_decoder.onnx")

# 入力名確認
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

print(f"   ONNX inputs: {input_names}")
print(f"   ONNX outputs: {output_names}")

# ONNX推論実行
onnx_inputs = {
    'projection_features': test_proj.numpy(),
    'global_embedding': test_global.numpy(),
    'view_dirs': test_dirs.numpy()
}

onnx_outputs = session.run(None, onnx_inputs)

# 出力解析
print("\n[3/3] Analyzing ONNX outputs...")

output_dict = dict(zip(output_names, onnx_outputs))

for name, data in output_dict.items():
    print(f"\n   {name}:")
    print(f"     Shape: {data.shape}")
    print(f"     Dtype: {data.dtype}")
    print(f"     Range: [{data.min():.6f}, {data.max():.6f}]")
    print(f"     Mean: {data.mean():.6f}")
    print(f"     Std: {data.std():.6f}")
    
    # サンプル値
    flat = data.flatten()
    sample_size = min(10, len(flat))
    print(f"     Sample: {flat[:sample_size]}")
    
    # 全て同じ値かチェック
    unique_count = len(np.unique(data))
    if unique_count == 1:
        print(f"     ⚠️  WARNING: All values are identical! ({data.flat[0]})")
    elif unique_count < 10:
        print(f"     ⚠️  WARNING: Only {unique_count} unique values")

# 特に重要な出力の詳細チェック
print("\n" + "=" * 70)
print("🔍 Critical Output Analysis")
print("=" * 70)

opacity = output_dict['opacity']
print(f"\n📊 Opacity Analysis:")
print(f"   Unique values: {len(np.unique(opacity))}")
if len(np.unique(opacity)) < 100:
    print(f"   Unique values list: {np.unique(opacity)[:20]}")
    
scale = output_dict['scale']
print(f"\n📊 Scale Analysis:")
print(f"   Per-component stats:")
for i in range(3):
    component = scale[:, :, i]
    print(f"     Dim {i}: min={component.min():.4f}, max={component.max():.4f}, mean={component.mean():.4f}")

rotation = output_dict['rotation']
print(f"\n📊 Rotation Analysis:")
print(f"   Normalized? {np.allclose(np.linalg.norm(rotation, axis=-1), 1.0)}")
print(f"   Norm stats: min={np.linalg.norm(rotation, axis=-1).min():.4f}, max={np.linalg.norm(rotation, axis=-1).max():.4f}")

id_emb = output_dict['id_embedding_256']
print(f"\n📊 ID Embedding Analysis:")
print(f"   Non-zeros: {np.count_nonzero(id_emb)} / {id_emb.size}")
print(f"   Range: [{id_emb.min():.4f}, {id_emb.max():.4f}]")

print("\n" + "=" * 70)
print("✅ Test Complete")
print("=" * 70)
