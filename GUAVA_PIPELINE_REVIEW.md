# GUAVA パイプライン実証テスト・再考察レポート

## 1. エグゼクティブサマリー

GUAVA論文 (arXiv:2505.03351) のPython公式実装と、TypeScript/WebGL再実装 (review-ply-generation-Nh8nJ) を比較検証した。
**5つの重大な不整合**と**3つの軽微な問題**を発見した。

| # | 問題 | 深刻度 | 状態 |
|---|------|--------|------|
| 1 | UV Pipeline: StyleUNetバイパス | **致命的** | 未修正 |
| 2 | DINOv2 feature map次元の混同 (f_map1 vs f_map2) | **致命的** | 未修正 |
| 3 | UV Decoder ONNX: 活性化関数の二重/欠落 | **重大** | 要検証 |
| 4 | UV Scale × 0.05 の余計な係数 | 中 | 修正済み (03ad9c0) |
| 5 | Refiner入力の不正な正規化 | 中 | 修正済み (03ad9c0) |
| 6 | Template色のグレー化 (pre-sigmoid mean≈0) | 軽微 | 仕様通り |
| 7 | GPU Device Lost (DXGI_ERROR_DEVICE_HUNG) | 軽微 | 環境依存 |
| 8 | UV解像度 1024×1024 vs Python 512×512 | 軽微 | 要確認 |

---

## 2. パイプライン比較：Python公式 vs TypeScript実装

### 2.1 全体アーキテクチャ（Python公式）

```
Input Image (518×518)
    ↓
DINOv2 (frozen, vitb14)
    ↓
DINO_Encoder (RefineNet + FPN)
    ↓
┌──────────────────────────────────┬──────────────────────────────────────────┐
│ f_map1: [B, 32ch, 512, 512]     │ f_map2: [B, 128ch, 512, 512]            │
│ (dino_out_dim=32)               │ (prj_out_dim=128)                       │
│ → UV Branch用                   │ → Projection Sampling用                 │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │                                          │
│ 1. cat(RGB(3ch), f_map1(32ch))  │ 1. sample_prj_feature()                 │
│    = 35ch                        │    → [B, N_vertex, 128]                  │
│                                  │                                          │
│ 2. inverse_texture_mapping()     │ 2. cat(projection(128), base(128),      │
│    → [B, 35ch, 512, 512]        │        global(256))                      │
│                                  │    → [B, N_vertex, 512]                  │
│ 3. StyleUNet(35ch → 96ch,       │                                          │
│    style=uv_style_mapping(768→512))│ 3. Vertex_GS_Decoder(512)             │
│    → [B, 96ch, 512, 512]        │    → colors(32ch), opacity, scale, rot  │
│                                  │                                          │
│ 4. cat(styleunet_out(96ch),      │                                          │
│        uv_base_feature(32ch))    │                                          │
│    = 128ch                       │                                          │
│                                  │                                          │
│ 5. UV_Point_GS_Decoder(128+27dir)│                                          │
│    → colors(32ch), opacity,      │                                          │
│      scale, rotation, local_pos  │                                          │
└──────────────────────────────────┴──────────────────────────────────────────┘
    ↓
Merge (Template + UV Gaussians)
    ↓
sigmoid(colors[:,:3])  ← 最初の3chのみsigmoid
    ↓
GaussianRasterizer_32 (32ch rendering)
    ↓
Neural Refiner (StyleUNet, 32ch → 3ch RGB)  ← 正規化なし
    ↓
Output RGB
```

### 2.2 TypeScript実装（review-ply-generation-Nh8nJ）

```
Input Image (518×518)
    ↓
DINOv2 (ONNX, vitb14)
    ↓
Encoder ONNX (768ch → 128ch)
    ↓
┌──────────────────────────────────┬──────────────────────────────────────────┐
│ appearanceMap: [128ch, 518, 518] │ projectionFeatures: [N, 128]            │
│ ★ f_map2相当 (128ch)            │ ← projection sampling with 128ch map   │
│                                  │                                          │
│ 1. UVFeatureMapper.mapToUV()     │ 1. Template Decoder (ONNX)              │
│    128ch → UV space              │    projection(128) + base(128) +        │
│    → [128ch, 1024, 1024]        │    global(256) → 512 → decoder          │
│                                  │    → latent32ch, opacity, scale, rot    │
│ 2. addViewEmbedding(27ch)        │                                          │
│    → [155ch, 1024, 1024]        │                                          │
│                                  │                                          │
│ 3. UV Point Decoder (ONNX)       │                                          │
│    155ch → decoder outputs       │                                          │
│    ★ StyleUNetを完全にバイパス   │                                          │
│    ★ uv_base_featureも欠落      │                                          │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 3. 重大な不整合の詳細分析

### 3.1 【致命的】UV Pipeline: StyleUNetバイパス

**Python公式パイプライン** (`ubody_gaussian.py:136-148`):
```python
# Step 1: RGB(3ch) + f_map1(32ch) = 35ch
img_feature = torch.cat([image_rgb, img_feature], dim=1)  # 35ch

# Step 2: Inverse texture mapping → UV space
uvmap_features = self.convert_pixel_feature_to_uv(img_feature, ...)  # [B, 35, 512, 512]

# Step 3: StyleUNet processing (CRITICAL!)
extra_style = self.uv_style_mapping(global_feature)  # 768 → 512
uvmap_features = self.uv_feature_decoder(uvmap_features, extra_style=extra_style)  # 35ch → 96ch

# Step 4: Concatenate learnable base features
uvmap_features = torch.cat([uvmap_features, self.uv_base_feature], dim=1)  # 96 + 32 = 128ch

# Step 5: UV decoder with cam_dirs
uv_point_gs_dict = self.uv_point_decoder(uvmap_features, cam_dirs)  # 128 + 27 = 155ch input
```

**TypeScript実装** (`gvrm.ts` review branch):
```typescript
// Step 1: Map 128ch appearance features to UV space (WRONG feature map!)
const uvFeatures128 = this.uvFeatureMapper.mapToUV(this.appearanceMap, ...);

// Step 2: Add view embedding → 155ch
const uvFeatures155 = this.uvFeatureMapper.addViewEmbedding(uvFeatures128, [0, 0, 1]);

// Step 3: UV Decoder (NO StyleUNet, NO base features!)
const uvGaussianOutput = await this.uvDecoder.generate(uvFeatures155, ...);
```

**問題点**:
1. **StyleUNetが完全に欠落**: Python側で35ch→96chの変換を行うStyleUNetがTS側に存在しない
2. **uv_base_feature(32ch)が欠落**: 学習可能なパラメータがTS側に渡されていない
3. **入力feature mapの混同**: f_map2(128ch, projection用)をf_map1(32ch, UV用)の代わりに使用
4. **入力次元の不一致**:
   - Python: UV_Point_GS_Decoder は `128ch + 27ch = 155ch` を期待
   - TS: `128ch(wrong feature) + 27ch = 155ch` → 次元は合うが中身が全く異なる

**影響**: UV Gaussianの品質が根本的に劣化。StyleUNetによる外観リファインメントが完全に欠如。

### 3.2 【致命的】DINOv2 Feature Map次元の混同

**Python** (`dino_encoder.py:62-63, config`):
```python
self.output_conv = nn.Conv2d(hidden_dims, output_dim, ...)    # f_map1: dino_out_dim=32ch
self.output_conv_2 = nn.Conv2d(hidden_dims, output_dim_2, ...) # f_map2: prj_out_dim=128ch
```

- `f_map1` (32ch): UV pipeline専用。RGB(3ch)と連結して35chとしてStyleUNet入力
- `f_map2` (128ch): Projection sampling専用。頂点ごとの特徴量抽出に使用

**TypeScript**: Encoder ONNXは128chを出力（`f_map2`相当）。これをProjection Sampling**と**UV Pipelineの両方に使用。

**問題**: f_map1(32ch)がTS側に存在しない。32chと128chは別のConv層で生成されており、特徴空間が異なる。

### 3.3 【重大】UV Decoder ONNX: 活性化関数の整合性

**Python** `UV_Point_GS_Decoder.forward()`:
```python
opacities = torch.sigmoid(opacities)   # line 123
scales = torch.exp(scales)             # line 126
rotations = F.normalize(rotations)     # line 129
colors = self.color_head(gaussian_feature)  # line 120 (NO activation)
```

**TypeScript** (`gvrm.ts` review branch):
```typescript
// ONNX出力後に手動で活性化関数を適用
uvOpacity[i] = 1.0 / (1.0 + Math.exp(-uvOpacity[i]));  // sigmoid
uvScale[i] = Math.exp(uvScale[i]);                       // exp
// rotation: normalize quaternion
```

**検証が必要な点**:
- ONNX export時にsigmoid/exp/normalizeが含まれているか？
- ログの `opacity range [-1.305, 0.365]` はsigmoid前 → ONNXにsigmoid未含 → TS側の手動適用は正しい
- ログの `scale range [-4.062, -0.309]` はexp前 → ONNXにexp未含 → TS側の手動適用は正しい
- **ただし**: ONNXモデルが `UV_Point_GS_Decoder` 単体であることが前提。もしStyleUNet込みのモデルなら入力仕様が異なる。

### 3.4 【修正済み】UV Scale × 0.05

**Python** (`feature_decoder.py:55` vs `126`):
- Template: `sigmoid(scales) * 0.05` ← 0.05はネットワーク設計の一部
- UV: `exp(scales)` ← 0.05なし

**PLY保存** (`ubody_gaussian.py:331-332`):
```python
scale_all_np = torch.log(self._uv_scaling_cano)
# _uv_scaling_cano = _uv_scaling * face_scaling_nn (line 283)
# _uv_scaling = exp(raw) (from UV decoder, line 126)
# So PLY = log(exp(raw) * face_scaling) = raw + log(face_scaling)
```

**結論**: PLYにface_scalingが焼き込まれている。`exp(PLY_value)` で正しい値に復元される。× 0.05は不要。
→ **コミット 03ad9c0 で正しく修正済み**

### 3.5 【修正済み】Refiner入力の不正な正規化

**Python** (`gaussian_render.py:73`):
```python
refine_images = self.nerual_refiner(rendered_images)  # 正規化なし
```

**TS（修正前）**:
```typescript
const normalizedFeatures = this.normalizeToZeroOne(coarseFeatures);
displayRGB = await this.neuralRefiner.process(normalizedFeatures);
```

→ **コミット 03ad9c0 で正しく修正済み**

---

## 4. 軽微な問題

### 4.1 Template色のグレー化

ログ: `Pre-sigmoid mean is near 0 (0.1364) → sigmoid will output ~0.5 (GRAY)`

これは**仕様通り**。Python側でも同様:
- `Vertex_GS_Decoder.forward()` line 49: `colors = self.color_layers(input_features)` — 生の32ch出力
- `Ubody_Gaussian.__init__()` line 186: `self._smplx_features_color[...,:3] = torch.sigmoid(...)` — 最初の3chのみsigmoid
- 残りの29chは生値のまま → Refinerが解釈する潜在特徴量
- 32ch全体のmean≈0は潜在特徴量として正常。最初の3ch（RGB）のみsigmoidで[0,1]に変換

### 4.2 GPU Device Lost

`DXGI_ERROR_DEVICE_HUNG (0x887A0006)` は、1,059,170個のGaussianをGPUでレンダリングする際のハードウェア制約。
TDR (Timeout Detection and Recovery) が発生する環境依存の問題。

### 4.3 UV解像度の不一致

- Python: `uvmap_size = 512` (config)
- TS: UV Triangle Mapping `1024×1024`、UV Decoder出力 `1024×1024`
- Python UV valid pixels ≈ 131,072 (512²の約50%)
- TS UV valid pixels = 1,048,575 (1024²のほぼ全て)

4倍の解像度差があるが、これは意図的な高解像度化かもしれない。ただしONNXモデルが512×512で学習されている場合、1024×1024入力は未テスト領域。

---

## 5. 修正アクションプラン

### Phase 1: 致命的問題の修正 (必須)

#### 1-A: f_map1 (32ch) の生成

**現状**: Encoder ONNXは128ch（f_map2相当）のみ出力。

**修正案**:
- Encoder ONNXを再export：`output_conv`（32ch）と`output_conv_2`（128ch）の両方を出力
- または: 別途 `dino_encoder_uv.onnx` を作成（32ch出力専用）
- **最小限の修正**: 既存128ch出力の最初の32chをf_map1の近似として使う（精度は低下）

#### 1-B: StyleUNet ONNXモデルの追加

**必要なモデル**: `styleunet_uv.onnx`
- 入力: `[B, 35, 512, 512]` (RGB + f_map1) + style `[B, 512]`
- 出力: `[B, 96, 512, 512]`

**修正手順**:
1. Python側で `uv_feature_decoder` (StyleUNet) をONNX exportする
2. `uv_style_mapping` (768→512 MLP) もONNX exportする
3. TS側で: f_map1(32ch) + RGB(3ch) → inverse texture mapping → StyleUNet → cat(base_feature) → UV Decoder

#### 1-C: uv_base_feature (32ch) の追加

**必要**: `uv_base_feature.bin` (32 × 512 × 512 = 16MB)

PyTorchから抽出:
```python
torch.save(model.uv_base_feature.data, 'uv_base_feature.bin')
```

TS側で: StyleUNet出力(96ch) + base_feature(32ch) = 128ch → UV Decoder

### Phase 2: 検証テスト

#### Test 1: Template Branch単体検証
- Python: 同一入力画像で `vertex_gs_dict` の出力値を記録
- TS: Template Decoder出力と数値比較
- 許容誤差: L1 < 0.01

#### Test 2: UV Branch入力検証
- Python: `uvmap_features`（StyleUNet出力 + base_feature）の統計値を記録
- TS: 修正後のUV decoder入力と比較

#### Test 3: End-to-End検証
- Python: `rendered_images`（32ch rasterizer出力）を保存
- TS: GPU splatting出力と比較
- Python: `refine_images`（Refiner出力RGB）を保存
- TS: Refiner出力と比較

#### Test 4: 活性化関数検証
- ONNX各モデルの出力に活性化関数が含まれているか確認
- UV Decoder: sigmoid(opacity), exp(scale), normalize(rotation) の有無

### Phase 3: 最適化 (後回し)

- UV解像度の統一（512 or 1024）
- GPU TDR対策（Gaussian pruning、タイルベースレンダリング）
- WebGPU compute shader最適化

---

## 6. Python側テストスクリプト（参考値取得用）

```python
# test_pipeline_values.py
# 各ステージの中間値を保存して、TS実装と比較する

import torch
import numpy as np
from models.UbodyAvatar.ubody_gaussian import Ubody_Gaussian_inferer

def extract_reference_values(model, batch):
    """各ステージの中間値を抽出"""

    # Step 1: DINOv2 features
    dino_dict = model.dino_encoder(batch['image'], output_size=model.cfg.image_size)
    f_map1 = dino_dict['f_map1']  # [B, 32, 512, 512]
    f_map2 = dino_dict['f_map2']  # [B, 128, 512, 512]
    f_global = dino_dict['f_global']  # [B, 768]

    print(f"f_map1: shape={f_map1.shape}, range=[{f_map1.min():.4f}, {f_map1.max():.4f}]")
    print(f"f_map2: shape={f_map2.shape}, range=[{f_map2.min():.4f}, {f_map2.max():.4f}]")
    print(f"f_global: shape={f_global.shape}, range=[{f_global.min():.4f}, {f_global.max():.4f}]")

    # Step 2: Global feature mapping
    vertex_global = model.global_feature_mapping(f_global)  # [B, 256]
    print(f"vertex_global: range=[{vertex_global.min():.4f}, {vertex_global.max():.4f}]")

    # Step 3: Projection sampling
    proj_feat, _ = model.sample_prj_feature(
        model.smplx_deform_res['vertices'], f_map2, batch['w2c_cam']
    )
    print(f"proj_feat: shape={proj_feat.shape}, range=[{proj_feat.min():.4f}, {proj_feat.max():.4f}]")

    # Step 4: Template decoder input
    vertex_input = torch.cat([proj_feat, model.vertex_base_feature[None],
                              vertex_global[:,None,:].expand(-1, proj_feat.shape[1], -1)], dim=-1)
    print(f"template_input: shape={vertex_input.shape}")

    # Step 5: UV pipeline
    image_rgb = torch.nn.functional.interpolate(batch['image'], (512, 512))
    img_feature = torch.cat([image_rgb, f_map1], dim=1)  # [B, 35, 512, 512]
    print(f"uv_input (pre-mapping): shape={img_feature.shape}")

    uvmap_features = model.convert_pixel_feature_to_uv(img_feature,
        model.smplx_deform_res['vertices'], batch['w2c_cam'])
    print(f"uvmap_features (post-mapping): shape={uvmap_features.shape}, range=[{uvmap_features.min():.4f}, {uvmap_features.max():.4f}]")

    extra_style = model.uv_style_mapping(f_global)  # [B, 512]
    uvmap_features = model.uv_feature_decoder(uvmap_features, extra_style=extra_style)
    print(f"uvmap_features (post-StyleUNet): shape={uvmap_features.shape}, range=[{uvmap_features.min():.4f}, {uvmap_features.max():.4f}]")

    uvmap_features = torch.cat([uvmap_features, model.uv_base_feature[None]], dim=1)
    print(f"uvmap_features (with base): shape={uvmap_features.shape}")

    # Save reference values
    np.savez('reference_values.npz',
        f_map1=f_map1[0].detach().cpu().numpy(),
        f_map2=f_map2[0].detach().cpu().numpy(),
        f_global=f_global[0].detach().cpu().numpy(),
        proj_feat=proj_feat[0].detach().cpu().numpy(),
        uvmap_pre_styleunet=uvmap_features[0].detach().cpu().numpy(),
    )
    print("Reference values saved to reference_values.npz")
```

---

## 7. 結論

TypeScript実装の最大の問題は、**UV PipelineでStyleUNetを完全にバイパスしている**ことである。
これはGUAVA論文のFig.2のアーキテクチャと根本的に異なる。

StyleUNetは入力画像の外観情報を**UV空間上でリファイン**する重要なコンポーネントであり、
これなしではUV Gaussianの品質が大幅に低下する。

修正の優先順位:
1. **StyleUNet ONNX export + TS統合** (最重要)
2. **f_map1 (32ch) の生成** (StyleUNetの正しい入力に必要)
3. **uv_base_feature のexport** (UV decoder入力の完全性)
4. End-to-end検証テスト

これらの修正により、UV Gaussianの品質がPython実装に近づき、
最終レンダリングの品質が大幅に改善されることが期待される。
