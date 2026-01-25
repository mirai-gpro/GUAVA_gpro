#!/usr/bin/env python3
"""
test_onnx_input_sensitivity.py

ONNXモデルが入力の変化に応じて出力が変わるかテスト
"""

import numpy as np
import onnxruntime as ort

print("=" * 70)
print("🧪 ONNX Input Sensitivity Test")
print("=" * 70)

# ONNX session
session = ort.InferenceSession("template_decoder.onnx")

N = 10595

print("\n[Test 1] Same input → Same output?")
print("-" * 70)

# Test 1: 同じ入力で2回実行
np.random.seed(42)
input1_proj = np.random.randn(1, N, 128).astype(np.float32)
input1_global = np.random.randn(1, 768).astype(np.float32)
input1_dirs = np.zeros((1, 27), dtype=np.float32)

result1 = session.run(None, {
    'projection_features': input1_proj,
    'global_embedding': input1_global,
    'view_dirs': input1_dirs
})

result1_again = session.run(None, {
    'projection_features': input1_proj,
    'global_embedding': input1_global,
    'view_dirs': input1_dirs
})

opacity1 = result1[1]
opacity1_again = result1_again[1]

same_output = np.allclose(opacity1, opacity1_again)
print(f"Same input → Same output: {same_output}")
if same_output:
    print("✅ Deterministic (good)")
else:
    print("❌ Non-deterministic (bad)")

print("\n[Test 2] Different projection_features → Different output?")
print("-" * 70)

# Test 2: projection_featuresだけ変える
np.random.seed(100)
input2_proj = np.random.randn(1, N, 128).astype(np.float32)  # 異なる値
input2_global = input1_global  # 同じ
input2_dirs = input1_dirs  # 同じ

result2 = session.run(None, {
    'projection_features': input2_proj,
    'global_embedding': input2_global,
    'view_dirs': input2_dirs
})

opacity2 = result2[1]

different_proj = not np.allclose(opacity1, opacity2)
print(f"Different projection → Different output: {different_proj}")
if different_proj:
    print("✅ Model uses projection_features (good)")
    max_diff = np.abs(opacity1 - opacity2).max()
    print(f"   Max difference: {max_diff:.6f}")
else:
    print("❌ Model IGNORES projection_features (BAD!)")
    print(f"   Opacity1 sample: {opacity1[0, :5, 0]}")
    print(f"   Opacity2 sample: {opacity2[0, :5, 0]}")

print("\n[Test 3] Different global_embedding → Different output?")
print("-" * 70)

# Test 3: global_embeddingだけ変える
np.random.seed(200)
input3_proj = input1_proj  # 同じ
input3_global = np.random.randn(1, 768).astype(np.float32)  # 異なる値
input3_dirs = input1_dirs  # 同じ

result3 = session.run(None, {
    'projection_features': input3_proj,
    'global_embedding': input3_global,
    'view_dirs': input3_dirs
})

opacity3 = result3[1]

different_global = not np.allclose(opacity1, opacity3)
print(f"Different global_embedding → Different output: {different_global}")
if different_global:
    print("✅ Model uses global_embedding (good)")
    max_diff = np.abs(opacity1 - opacity3).max()
    print(f"   Max difference: {max_diff:.6f}")
else:
    print("❌ Model IGNORES global_embedding (BAD!)")
    print(f"   Opacity1 sample: {opacity1[0, :5, 0]}")
    print(f"   Opacity3 sample: {opacity3[0, :5, 0]}")

print("\n[Test 4] All zeros input → What output?")
print("-" * 70)

# Test 4: ゼロ入力
zero_proj = np.zeros((1, N, 128), dtype=np.float32)
zero_global = np.zeros((1, 768), dtype=np.float32)
zero_dirs = np.zeros((1, 27), dtype=np.float32)

result_zero = session.run(None, {
    'projection_features': zero_proj,
    'global_embedding': zero_global,
    'view_dirs': zero_dirs
})

opacity_zero = result_zero[1]

print(f"Zero input opacity:")
print(f"   Min: {opacity_zero.min():.6f}")
print(f"   Max: {opacity_zero.max():.6f}")
print(f"   Mean: {opacity_zero.mean():.6f}")
print(f"   Unique values: {len(np.unique(opacity_zero))}")

if len(np.unique(opacity_zero)) == 1:
    print(f"   ⚠️  All same value: {opacity_zero.flat[0]:.6f}")
    print("   This suggests model might be using only base_features!")

print("\n[Test 5] Output statistics comparison")
print("-" * 70)

outputs = [
    ("Random 1", result1),
    ("Random 2 (diff proj)", result2),
    ("Random 3 (diff global)", result3),
    ("Zero input", result_zero)
]

print(f"{'Input':<25} {'Opacity':<30} {'Scale':<30} {'Rotation':<30}")
print("-" * 115)

for name, result in outputs:
    opacity = result[1]
    scale = result[2]
    rotation = result[3]
    
    op_unique = len(np.unique(opacity))
    sc_unique = len(np.unique(scale))
    rot_unique = len(np.unique(rotation))
    
    print(f"{name:<25} "
          f"[{opacity.min():.4f}, {opacity.max():.4f}] u={op_unique:<6} "
          f"[{scale.min():.4f}, {scale.max():.4f}] u={sc_unique:<6} "
          f"[{rotation.min():.4f}, {rotation.max():.4f}] u={rot_unique:<6}")

print("\n" + "=" * 70)
print("🔍 Analysis")
print("=" * 70)

if not different_proj and not different_global:
    print("❌ CRITICAL: Model IGNORES both projection and global inputs!")
    print("   The model is only using vertex_base_feature.")
    print("   This explains why all outputs are identical.")
    print("\n   Possible causes:")
    print("   1. ONNX export incorrectly handled input concatenation")
    print("   2. vertex_base_feature[:N, :] slicing not working in ONNX")
    print("   3. Feature fusion (torch.cat) not exported correctly")
elif not different_proj:
    print("❌ Model IGNORES projection_features")
elif not different_global:
    print("❌ Model IGNORES global_embedding")
else:
    print("✅ Model correctly uses both inputs")

print("\n" + "=" * 70)
