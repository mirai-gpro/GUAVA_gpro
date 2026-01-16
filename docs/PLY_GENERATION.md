# GUAVA PLY File Generation Guide

論文「GUAVA: Generalizable Upper Body 3D Gaussian Avatar」に準拠したPLYファイルを生成するためのガイドです。

## 出力PLYファイル形式

### 1. GS_canonical.ply（3D Gaussian Splatting形式）

論文のUbody Gaussians形式に準拠した3DGS PLYファイル：

| 属性 | 型 | 説明 |
|------|------|------|
| x, y, z | float32 | 位置（canonical空間） |
| nx, ny, nz | float32 | 法線（ゼロ初期化） |
| f_dc_0, f_dc_1, f_dc_2 | float32 | SH係数（RGB→SH変換済み） |
| opacity | float32 | 不透明度（inverse sigmoid） |
| scale_0, scale_1, scale_2 | float32 | スケール（log scale） |
| rot_0, rot_1, rot_2, rot_3 | float32 | 回転クォータニオン（wxyz順） |

### 2. canonical.ply（Open3D点群形式）

シンプルな点群形式：
- 位置: x, y, z
- 色: RGB

### 3. 分割ファイル

`save_split=True`の場合、以下も生成：
- `GS_canonical_smplx.ply`: Template Gaussians（SMPLX頂点ベース）
- `GS_canonical_uv.ply`: UV Gaussians（UVマップベース）

---

## 使用方法

### Modal（クラウドGPU）での実行

```bash
# 基本使用法（EHM-Trackerの結果を自動検出）
modal run generate_ply_modal.py

# データパスを指定
modal run generate_ply_modal.py --data-path /path/to/tracked/data

# 出力名を指定
modal run generate_ply_modal.py --output-name my_avatar

# Gaussian PLYのみ生成
modal run generate_ply_modal.py --gaussian-only

# 分割保存なし
modal run generate_ply_modal.py --split False
```

### ローカル（Anaconda）での実行

```bash
# 環境アクティベート
conda activate GUAVA

# 基本使用法
python generate_ply_local.py --data_path assets/example/tracked_video/xxx

# 出力先を指定
python generate_ply_local.py --data_path <path> --output_dir outputs/ply

# Gaussian PLYのみ
python generate_ply_local.py --data_path <path> --gaussian_only

# 点群PLYのみ
python generate_ply_local.py --data_path <path> --point_cloud_only

# Deformed PLY（ポーズ適用後）も保存
python generate_ply_local.py --data_path <path> --with_deformed

# PLY形式の説明を表示
python generate_ply_local.py --format_info
```

---

## 前提条件

### 1. トラッキングデータ

EHM-Trackerでトラッキング済みのデータが必要です。

```bash
# EHM-Trackerでのトラッキング例
cd EHM-Tracker
python tracking_video.py --in-root ./data/videos --output-dir ./output/processed_data
```

### 2. モデルウェイト

GUAVAの事前学習済みウェイトが必要です：
- `assets/GUAVA/checkpoints/best.pt` または `latest.pt`

### 3. パラメトリックモデル

- SMPLX: `assets/SMPLX/SMPLX_NEUTRAL_2020.npz`
- FLAME: `assets/FLAME/FLAME2020/generic_model.pkl`

---

## 出力例

```
ply_output/
├── video_id_001/
│   ├── canonical.ply           # 点群（位置+色）
│   ├── canonical_smplx.ply     # SMPLX頂点の点群
│   ├── canonical_uv.ply        # UV Gaussiansの点群
│   ├── GS_canonical.ply        # 3DGS形式（全体）
│   ├── GS_canonical_smplx.ply  # 3DGS形式（Template）
│   ├── GS_canonical_uv.ply     # 3DGS形式（UV）
│   └── deformed.ply            # ポーズ適用後（オプション）
└── generation_summary.json     # 生成情報
```

---

## PLYファイルの使用

### 3D Gaussian Splattingビューア

```python
# 3DGS標準ビューアで表示可能
# https://github.com/graphdeco-inria/gaussian-splatting
```

### Open3Dで読み込み

```python
import open3d as o3d

# 点群読み込み
pcd = o3d.io.read_point_cloud("canonical.ply")
o3d.visualization.draw_geometries([pcd])
```

### plyfileで属性確認

```python
from plyfile import PlyData

plydata = PlyData.read("GS_canonical.ply")
vertex = plydata['vertex']
print(f"Gaussian数: {len(vertex)}")
print(f"属性: {vertex.dtype.names}")
```

---

## トラブルシューティング

### 1. CUDAメモリエラー

```bash
# バッチサイズを減らすか、GPUメモリの大きいインスタンスを使用
# Modal: --gpu a100 オプションを検討
```

### 2. チェックポイントが見つからない

```bash
# assets/GUAVA/checkpoints/ にモデルウェイトを配置
bash assets/Docs/run_download.sh
```

### 3. YAMLパスエラー

Windows環境でアセットをコピーした場合、パスの修正が必要な場合があります。
スクリプトは自動的にパスを修正しますが、手動で修正する場合：

```yaml
# assets/GUAVA/config.yaml
# C:\path\to... → ./assets/... に変更
```

---

## 参考

- 論文: [GUAVA: Generalizable Upper Body 3D Gaussian Avatar](https://arxiv.org/abs/2505.03351)
- プロジェクトページ: https://eastbeanzhang.github.io/GUAVA/
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
