# 論文 vs TypeScript実装 詳細比較

## 🔴 重大な差異（テクスチャ再現に直接影響）

---

### 1. DINO Encoder 出力チャンネル

| 項目 | 論文/Python | TypeScript | 影響度 |
|------|-------------|------------|--------|
| UV branch | **160ch** (専用出力層 `f_map1`) | **32ch** (128chのサブセット) | 🔴 重大 |
| Template branch | **128ch** (専用出力層 `f_map2`) | **128ch** | ✅ OK |
| Global feature | **768ch** (CLS token) | **768ch** | ✅ OK |

**Python (dino_encoder.py:62-63):**
```python
self.output_conv = nn.Conv2d(hidden_dims, output_dim, ...)      # 32ch (UV) - 実際は160ch設定
self.output_conv_2 = nn.Conv2d(hidden_dims, output_dim_2, ...)  # 128ch (Template)

return {'f_map1': out, 'f_map2': out_2, 'f_global': out_global}
```

**TypeScript (image-encoder.ts:348-376):**
```typescript
if (appearanceChannels >= 128 + 32) {
  // 160ch → 128ch(Template) + 32ch(UV) として分離
  for (let c = 0; c < 32; c++) {
    this.uvFeatures[i * 32 + c] = appearanceData[(128 + c) * numPixels + i];  // ch 128-159
  }
} else if (appearanceChannels === 128) {
  // ⚠️ 128chしかない場合、最初の32chをUV用に流用
  for (let c = 0; c < 32; c++) {
    this.uvFeatures[i * 32 + c] = appearanceData[c * numPixels + i];  // ch 0-31
  }
}
```

**問題点:**
- 論文では `dino_out_dim=32` だが、実際の `f_map1` は **160ch** (config確認必要)
- TypeScript は **32ch** しか使用していない
- UV branch の情報量が **5倍不足**

---

### 2. Inverse Texture Mapping の入力

| 項目 | 論文/Python | TypeScript | 影響度 |
|------|-------------|------------|--------|
| 入力 | **RGB(3ch) + f_map1(160ch) = 163ch** | **f_uv(32ch)のみ** | 🔴 重大 |
| RGB統合 | 一緒にマッピング | 別々にマッピング | 🟡 中程度 |

**Python (ubody_gaussian.py:137-141):**
```python
# RGB と features を結合してから一緒にマッピング
image_rgb = nn.functional.interpolate(batch['image'], (self.cfg.image_size, self.cfg.image_size))
img_feature = torch.cat([image_rgb, img_feature], dim=1)  # [B, 3+160, H, W] = 163ch

# 163ch を UV空間にマッピング
uvmap_features = self.convert_pixel_feature_to_uv(img_feature, deformed_vertices, w2c_cam)
```

**TypeScript (gvrm.ts):**
```typescript
// 32ch features と 3ch RGB を別々にマッピング
const uvSpaceFeatures1024 = this.inverseMapper.map(..., imageBranchFeatures, ..., 32);  // 32ch
const uvSpaceRGB1024 = this.inverseMapper.map(..., sourceRGB.data, ..., 3);             // 3ch
```

**問題点:**
- Python は **163ch** を一緒に投影
- TypeScript は **32ch + 3ch = 35ch** を別々に投影
- 情報量が **約5倍不足** (163ch vs 35ch)

---

### 3. Inverse Texture Mapping の処理方法

| 項目 | 論文/Python | TypeScript | 影響度 |
|------|-------------|------------|--------|
| UV→3D変換 | `uvmap_f_idx` + `uvmap_f_bary` (重心座標補間) | `worldPositions` (事前計算) | 🟡 中程度 |
| 可視性マスク | `visible_faces` で隠面処理 | **なし** | 🔴 重大 |
| マスク適用 | `uvmap_mask` | `uvmap_mask` (類似) | ✅ OK |

**Python (ubody_gaussian.py:85-118):**
```python
def convert_pixel_feature_to_uv(self, img_features, deformed_vertices, w2c_cam, visble_faces=None):
    # 1. UV座標 → 三角形ID + 重心座標
    uvmap_f_idx = self.smplx.uvmap_f_idx       # [H, W]
    uvmap_f_bary = self.smplx.uvmap_f_bary     # [H, W, 3]

    # 2. 重心座標で3D位置を補間
    uv_vertex = torch.einsum('bhwk,bhwkn->bhwn', uvmap_f_bary, face_vertices)

    # 3. 3D → Camera → Image 投影
    uv_vertex_cam = torch.einsum('bij,bhwj->bhwi', w2c_cam, uv_vertex_homo)
    vertices_img = uv_vertex_cam * invtanfov / (z + eps)

    # 4. grid_sample
    uv_features = nn.functional.grid_sample(img_features, vertices_img[:,:,:,:2])

    # 5. ⚠️ 可視性マスク適用 (occluded faces を除外)
    if visble_faces is not None:
        visble_mask = torch.isin(all_faces, torch.unique(visble_faces))
        mask = mask * visble_mask

    uv_features = uv_features * mask
```

**TypeScript (inverse-texture-mapping.ts):**
```typescript
// worldPositions は事前計算済み（重心座標補間は実行時に行わない）
// visible_faces による隠面処理は実装されていない
sampleFeaturesSimplified(uvMapping, features, featureSize, featureChannels) {
    // worldPositions → Project to screen → Sample features
    // ⚠️ 可視性チェックなし
}
```

**問題点:**
- TypeScript は**隠面処理がない**ため、オクルージョンされた部分の特徴も混入
- 動的な頂点変形に対応していない（事前計算された worldPositions を使用）

---

### 4. StyleUNet 入力

| 項目 | 論文/Python | TypeScript | 影響度 |
|------|-------------|------------|--------|
| 入力チャンネル | **163ch** (RGB 3 + features 160) | **35ch** (RGB 3 + features 32) | 🔴 重大 |
| extra_style | 768ch → 512ch mapping | 768ch → 512ch mapping | ✅ OK |

**Python (ubody_gaussian.py:144-146):**
```python
# 入力: 163ch (RGB + DINO features)
extra_style = self.uv_style_mapping(global_feature)  # 768 → 512
uvmap_features = self.uv_feature_decoder(uvmap_features, extra_style=extra_style)  # 163ch → 96ch
```

**TypeScript (uv-styleunet.ts):**
```typescript
// 入力: 35ch (RGB 3 + UV features 32)
// light_styleunet_fp16.onnx は 35ch 入力で学習されている
const output = await this.session.run({
  'uv_features': [1, 35, H, W],
  'extra_style': [1, 512]
});
```

**問題点:**
- 論文は **163ch** 入力 → TypeScript は **35ch** 入力
- モデル自体が異なる入力仕様で学習されている可能性

---

### 5. UV Decoder 入力

| 項目 | 論文/Python | TypeScript | 影響度 |
|------|-------------|------------|--------|
| StyleUNet出力 | **96ch** | **96ch** | ✅ OK |
| base_feature | **32ch** | **32ch** | ✅ OK |
| view_dirs | **27ch** (内部で結合) | **27ch** (外部で結合) | ✅ OK |
| 合計 | **128ch + 27ch = 155ch** | **96ch + 32ch + 27ch = 155ch** | ✅ OK |

**Python (ubody_gaussian.py:147-153):**
```python
# StyleUNet出力(96ch) + base_feature(32ch) = 128ch
uvmap_features = torch.cat([uvmap_features, self.uv_base_feature], dim=1)

# UV Point Decoder内部で view_dirs (27ch) を結合
# 入力: 128ch、view_dirs は内部で追加
uv_point_gs_dict = self.uv_point_decoder(uvmap_features, cam_dirs)
```

**TypeScript (gvrm.ts:787-794):**
```typescript
// StyleUNet出力(96ch) + base_feature(32ch) = 128ch
const features128ch = this.uvStyleUNet.addBaseFeature(styleunetOutput, ...);

// view_dirs (27ch) を外部で結合
// 128ch + 27ch = 155ch
uvFeatureMap = concatenateWithViewEncoding(features128ch, viewDir, ...);
```

**差異:** view_dirs の結合タイミングが異なるが、最終的に同じ 155ch になるため問題なし

---

### 6. Gaussian Splatting

| 項目 | 論文/Python (CUDA) | TypeScript | 影響度 |
|------|-------------------|------------|--------|
| アルゴリズム | **EWA Splatting** (diff-gaussian-rasterization-32) | EWA Splatting (実装済み) | ✅ OK |
| 共分散計算 | 3D→2D covariance projection | 3D→2D covariance projection | ✅ OK |
| Gaussian形状 | **異方性楕円** (conic matrix) | 異方性楕円 (conic matrix) | ✅ OK |

**最近の修正で対応済み**

---

### 7. Neural Refiner

| 項目 | 論文/Python | TypeScript | 影響度 |
|------|-------------|------------|--------|
| 入力 | **32ch** latent features | **32ch** latent features | ✅ OK |
| 出力 | **3ch** RGB | **3ch** RGB | ✅ OK |
| 活性化 | clamp [0,1] | sigmoid (条件付き) + clamp | 🟡 中程度 |

**Python:**
```python
# SimpleUNet: 32ch → 3ch, 出力はそのまま clamp
```

**TypeScript (rfdn-refiner-webgpu.ts:185-205):**
```typescript
// raw output の範囲をチェック
const useSigmoid = rawMin < -1 || rawMax > 2;  // pre-sigmoid値っぽい場合

if (useSigmoid) {
  val = 1 / (1 + Math.exp(-val));  // sigmoid適用
}
val = Math.max(0, Math.min(1, val));  // clamp
```

**問題点:**
- sigmoid適用の判断ロジックが不明確
- モデルの学習時の出力仕様と不一致の可能性

---

## 📊 差異まとめ

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        論文 vs TypeScript 差異マップ                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DINO Encoder                                                           │
│                                                                                 │
│   論文:  f_map1 (160ch, UV用) + f_map2 (128ch, Template用)                      │
│   TS:    160ch → 128ch (Template) + 32ch (UV, サブセット)                       │
│                                                                                 │
│   ❌ UV branch のチャンネル数: 160ch → 32ch (情報量 1/5)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Inverse Texture Mapping                                                │
│                                                                                 │
│   論文:  RGB(3ch) + f_map1(160ch) = 163ch → UV空間                             │
│          + visible_faces による隠面処理                                         │
│   TS:    f_uv(32ch) → UV空間, RGB(3ch) → UV空間 (別々)                         │
│          隠面処理なし                                                           │
│                                                                                 │
│   ❌ 入力チャンネル: 163ch → 35ch (情報量 1/5)                                  │
│   ❌ 可視性マスク: あり → なし                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: StyleUNet                                                              │
│                                                                                 │
│   論文:  163ch → 96ch                                                          │
│   TS:    35ch → 96ch  (light_styleunet_fp16.onnx)                              │
│                                                                                 │
│   ⚠️ モデルが異なる入力仕様で学習されている可能性                               │
│   ⚠️ 情報量不足のため、96ch出力の質が低下                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4-5: UV Decoder + Template Decoder                                        │
│                                                                                 │
│   ✅ 入出力仕様は一致                                                           │
│   ⚠️ ただし、入力の質が低いため出力も低品質                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Gaussian Splatting                                                     │
│                                                                                 │
│   ✅ EWA Splatting 実装済み (最近修正)                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Neural Refiner                                                         │
│                                                                                 │
│   ✅ 入出力仕様は一致                                                           │
│   ⚠️ 入力 latent32ch の質が低いため、出力RGBも低品質                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 必要な修正

### 優先度1: DINO Encoder 出力の修正

```typescript
// 現状: 32ch (サブセット)
// 目標: 160ch (専用出力)

// Option A: dino_encoder.onnx を 160ch 出力に再エクスポート
// Option B: TypeScript で 160ch 全てを UV branch に使用
```

### 優先度2: Inverse Texture Mapping の修正

```typescript
// 現状: RGB(3ch) + features(32ch) = 35ch, 隠面処理なし
// 目標: RGB(3ch) + features(160ch) = 163ch, 隠面処理あり

// 必要なデータ:
// - uvmap_f_idx: [512, 512] 各UVピクセルの三角形ID
// - uvmap_f_bary: [512, 512, 3] 重心座標
// - visible_faces: レンダリング時に見える三角形のリスト
```

### 優先度3: StyleUNet モデルの再学習

```
現状の light_styleunet_fp16.onnx は 35ch 入力で学習されている
163ch 入力に対応するモデルが必要
```

---

## 📁 関連ファイル

| ファイル | 論文対応セクション | 主要な差異 |
|----------|-------------------|-----------|
| `image-encoder.ts` | DINO Encoder | UV branch 160ch → 32ch |
| `inverse-texture-mapping.ts` | Inverse Texture Mapping | 163ch → 35ch, 隠面処理なし |
| `uv-styleunet.ts` | StyleUNet | 163ch → 35ch 入力 |
| `uv-decoder.ts` | UV Point Decoder | ✅ OK |
| `template-decoder.ts` | Template Decoder | ✅ OK |
| `gaussian-rasterizer-webgpu.ts` | Gaussian Rendering | ✅ OK (修正済み) |
| `rfdn-refiner-webgpu.ts` | Neural Refiner | ⚠️ sigmoid判定ロジック |
