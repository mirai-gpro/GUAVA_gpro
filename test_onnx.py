# 実際の入力を使用
import numpy as np

# ダミーではなく、実際の値で
real_proj = np.random.randn(1, 10595, 128).astype(np.float32)
real_global = np.random.randn(1, 768).astype(np.float32)
real_dirs = np.zeros((1, 27), dtype=np.float32)

# 2回実行して同じ結果か確認
result1 = session.run(None, {
    'projection_features': real_proj,
    'global_embedding': real_global,
    'view_dirs': real_dirs
})

# 異なる入力
different_proj = np.random.randn(1, 10595, 128).astype(np.float32) 
different_global = np.random.randn(1, 768).astype(np.float32)

result2 = session.run(None, {
    'projection_features': different_proj,
    'global_embedding': different_global,
    'view_dirs': real_dirs
})

# 結果が同じか確認
print("Same opacity?", np.allclose(result1[1], result2[1]))