# GUAVA Pipeline 修正ガイド

## 公式実装の正確な理解

### 全体アーキテクチャ (論文 Fig. 2 参照)

```
Input Image (518x518)
    ↓
DINOv2 (frozen)
    ↓
[CLS token: 1024d] + [Patch tokens: 1369×1024 = 37×37 patches]
    ↓
┌─────────────────────────────┬──────────────────────────────┐
│ Template Gaussian Branch    │ UV Gaussian Branch           │
├─────────────────────────────┼──────────────────────────────┤
│ 1. Conv Encoder             │ 1. Conv Encoder              │
│    1024ch → 160ch (37×37)   │    1024ch → 160ch (37×37)    │
│                             │                              │
│ 2. Projection Sampling      │ 2. Inverse Texture Mapping   │
│    Vertices → Image         │    Image → UV Space          │
│    (canonical camera)       │    (canonical camera)        │
│                             │                              │
│ 3. Template Decoder         │ 3. StyleUNet + UV Decoder    │
│    Input: 128ch + 128ch     │    Input: 160ch + RGB + 32ch │
│           (proj + base)     │                              │
│    Output: latent32, opa,   │    Output: latent32, opa,    │
│            scale, rotation  │            scale, rotation   │
└─────────────────────────────┴──────────────────────────────┘
```

## 重要な実装の詳細

### 1. Camera Parameters (data_loader.py)

```python
# Canonical camera (常に固定)
w2c_cam = [[1, 0, 0,    0],
           [0, 1, 0,  0.6],
           [0, 0, 1,   22],
           [0, 0, 0,    1]]

tanfov = 1.0 / 24.0
invtanfov = 24.0  # focal length相当
```

**重要**: Source imageは常にcanonical viewを使用。Target imageは異なるカメラパラメータを持つ。

### 2. Image Encoder (ubody_gaussian.py lines 120-122)

```python
dino_feature_dict = self.dino_encoder(batch['image'], output_size=self.cfg.image_size)
img_feature = dino_feature_dict['f_map1']      # 160ch, 37×37 (for UV)
img_feature_2 = dino_feature_dict['f_map2']    # 128ch, 37×37 (for projection)
global_feature = dino_feature_dict['f_global'] # CLS token
```

**出力**:
- `f_map1`: 160 channels (UV branch用)
- `f_map2`: 128 channels (Projection sampling用)
- `f_global`: CLS token → 256d ID embedding

### 3. Projection Sampling (lines 75-83)

```python
def sample_prj_feature(self, vertices, feature_map, w2c_cam, vertices_img=None):
    # Step 1: World to Camera
    vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:,:,:1])], dim=-1)
    vertices_cam = torch.einsum('bij,bnj->bni', w2c_cam, vertices_homo)[:,:,:3]
    
    # Step 2: Perspective Projection
    vertices_img = vertices_cam * self.cfg.invtanfov / (vertices_cam[:,:,2:] + 1e-7)
    
    # Step 3: Grid Sample (align_corners=False)
    sampled_features = nn.functional.grid_sample(
        feature_map, 
        vertices_img[:,None,:,:2],  # [B, 1, N, 2]
        mode='bilinear', 
        padding_mode='border', 
        align_corners=False
    )
    return sampled_features.squeeze(-2).permute(0,2,1)  # [B, N, C]
```

**重要**: 
- `align_corners=False`: grid [-1, 1] → pixel [0, W-1]
- Formula: `pixel = ((grid + 1) * W - 1) / 2`

### 4. Inverse Texture Mapping (lines 85-114)

```python
def convert_pixel_feature_to_uv(self, img_features, deformed_vertices, w2c_cam, 
                                visble_faces=None, uv_features=None):
    # 1. UV map上の各ピクセルが属する三角形を取得
    uvmap_f_idx = self.smplx.uvmap_f_idx       # [H, W] triangle indices
    uvmap_f_bary = self.smplx.uvmap_f_bary     # [H, W, 3] barycentric coords
    
    # 2. 重心座標で3D位置を補間
    uv_vertex_id = faces[uvmap_f_idx]          # [H, W, 3] vertex indices
    uv_vertex = deformed_vertices[..., uv_vertex_id, :]  # [B, H, W, 3, 3]
    uv_vertex = torch.einsum('bhwk,bhwkn->bhwn', uvmap_f_bary, uv_vertex)  # [B, H, W, 3]
    
    # 3. 3D → Camera space → Image space
    uv_vertex_homo = torch.cat([uv_vertex, torch.ones_like(uv_vertex[:,:,:,:1])], dim=-1)
    uv_vertex_cam = torch.einsum('bij,bhwj->bhwi', w2c_cam, uv_vertex_homo)[:,:,:,:3]
    vertices_img = uv_vertex_cam * self.cfg.invtanfov / (uv_vertex_cam[...,2:] + 1e-7)
    
    # 4. Grid sample (Image → UV)
    uv_features = nn.functional.grid_sample(
        img_features,               # [B, C, H_img, W_img]
        vertices_img[:,:,:,:2],     # [B, H_uv, W_uv, 2]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # 5. Apply UV mask (valid region only)
    uv_features = uv_features * self.smplx.uvmap_mask[None, None]
    
    return uv_features  # [B, C, H_uv, W_uv]
```

**処理フロー**:
1. UV map → 3D mesh (barycentric interpolation)
2. 3D mesh → Camera space (world-to-camera transform)
3. Camera space → Image space (perspective projection)
4. Image features → UV features (grid sampling)
5. Mask application (invalid regions = 0)

### 5. UV Decoder (lines 136-148)

```python
# Prepare UV input
image_rgb = nn.functional.interpolate(batch['image'], 
                                     (self.cfg.image_size, self.cfg.image_size))
img_feature = torch.cat([image_rgb, img_feature], dim=1)  # [B, 3+160, H, W]

# Inverse texture mapping
uvmap_features = self.convert_pixel_feature_to_uv(img_feature, 
                                                   deformed_vertices, 
                                                   w2c_cam)  # [B, 163, 512, 512]

# StyleUNet processing
extra_style = self.uv_style_mapping(global_feature)  # [B, 512]
uvmap_features = self.uv_feature_decoder(uvmap_features, extra_style=extra_style)

# Concatenate base features
uvmap_features = torch.cat([uvmap_features, 
                            self.uv_base_feature[None].expand(batch_size,-1,-1,-1)], 
                           dim=1)

# UV Point Decoder
uv_point_gs_dict = self.uv_point_decoder(uvmap_features, cam_dirs)
```

## TypeScript実装の修正ポイント

### 修正1: image-encoder.ts

**問題点**:
- Canonical cameraパラメータが不正確
- Projection samplingのalign_corners処理が間違っている
- Feature dimensionsが公式と不一致

**修正版**: `/mnt/user-data/outputs/image-encoder-corrected.ts`

主な変更:
```typescript
// Before
const CONST_TX = 0.0;
const CONST_TY = 0.6;
const CONST_TZ = 22.0;

// After (明示的な構造化)
const CANONICAL_W2C = {
    R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T: [0.0, 0.6, 22.0]
};

// Before (間違ったalign_corners処理)
const u = ((ndcX + 1.0) * mapSize - 1.0) / 2.0;

// After (正しいPyTorch grid_sample互換)
const u = ((imgX + 1.0) * mapSize - 1.0) / 2.0;
```

### 修正2: inverse-texture-mapping.ts

**必要な修正**:
1. Barycentric interpolationの実装
2. 正しいtransform chain (UV → 3D → Camera → Image)
3. Visibility maskingの実装

**参考実装**:
```typescript
class InverseTextureMapper {
    // EHM mesh dataが必要:
    // - uvmap_f_idx: [H, W] 各UVピクセルが属する三角形ID
    // - uvmap_f_bary: [H, W, 3] 重心座標
    // - uvmap_mask: [H, W] 有効領域マスク
    // - faces: [F, 3] 三角形の頂点インデックス
    
    mapImageToUV(
        imageFeatures: Float32Array,  // [C, H_img, W_img]
        vertices: Float32Array,        // [V, 3] deformed vertices
        w2c: Float32Array,             // [4, 4] world-to-camera
        uvmapSize: number              // 512
    ): Float32Array {
        const output = new Float32Array(C * uvmapSize * uvmapSize);
        
        for (let v = 0; v < uvmapSize; v++) {
            for (let u = 0; u < uvmapSize; u++) {
                const idx = v * uvmapSize + u;
                
                // 1. Get triangle and barycentric coords
                const faceIdx = this.uvmap_f_idx[idx];
                const bary = this.uvmap_f_bary.slice(idx * 3, idx * 3 + 3);
                
                if (faceIdx < 0) continue;  // Invalid pixel
                
                // 2. Get triangle vertices
                const v0 = this.faces[faceIdx * 3 + 0];
                const v1 = this.faces[faceIdx * 3 + 1];
                const v2 = this.faces[faceIdx * 3 + 2];
                
                // 3. Interpolate 3D position
                const pos = [0, 0, 0];
                for (let d = 0; d < 3; d++) {
                    pos[d] = bary[0] * vertices[v0 * 3 + d] +
                             bary[1] * vertices[v1 * 3 + d] +
                             bary[2] * vertices[v2 * 3 + d];
                }
                
                // 4. Transform to camera space
                const camPos = transformPoint(pos, w2c);
                
                // 5. Project to image
                const imgX = (camPos[0] * INV_TAN_FOV) / camPos[2];
                const imgY = (camPos[1] * INV_TAN_FOV) / camPos[2];
                
                // 6. Sample image features (bilinear)
                const sampledFeatures = sampleBilinear(
                    imageFeatures, imgX, imgY, H_img, W_img, C
                );
                
                // 7. Write to UV map
                for (let c = 0; c < C; c++) {
                    output[c * uvmapSize * uvmapSize + idx] = sampledFeatures[c];
                }
            }
        }
        
        // Apply mask
        for (let i = 0; i < uvmapSize * uvmapSize; i++) {
            if (!this.uvmap_mask[i]) {
                for (let c = 0; c < C; c++) {
                    output[c * uvmapSize * uvmapSize + i] = 0;
                }
            }
        }
        
        return output;
    }
}
```

### 修正3: template-decoder.ts

**必要な入力**:
```typescript
interface TemplateDecoderInput {
    projection_feature: Float32Array;  // [N, 128] from projection sampling
    base_feature: Float32Array;        // [N, 128] learnable vertex features
    id_embedding: Float32Array;        // [256] from CLS token
}

interface TemplateDecoderOutput {
    latent_32ch: Float32Array;  // [N, 32] latent features
    opacity: Float32Array;      // [N] inverse sigmoid space
    scale: Float32Array;        // [N, 3] log space
    rotation: Float32Array;     // [N, 4] quaternion
}
```

### 修正4: uv-decoder.ts

**必要な入力**:
```typescript
interface UVDecoderInput {
    uv_features: Float32Array;    // [C, 512, 512] from inverse mapping
    base_features: Float32Array;  // [32, 512, 512] learnable
    style_embedding: Float32Array; // [512] from global feature
}
```

## デバッグチェックリスト

### Phase 1: Image Encoder
- [ ] DINOv2出力が正しい (1370 tokens, 1024 dim)
- [ ] CLS tokenとpatch tokensが正しく分離
- [ ] Encoder出力が160 channels
- [ ] Projection sampling結果がゼロでない
- [ ] ID embeddingが256次元

### Phase 2: Inverse Texture Mapping
- [ ] UV map sizeが512×512
- [ ] Barycentric interpolationが正しい
- [ ] 3D → Camera → Image transformが正確
- [ ] Grid samplingがalign_corners=Falseに準拠
- [ ] Maskが正しく適用される

### Phase 3: Decoders
- [ ] Template Decoder入力: projection(128) + base(128) + id(256)
- [ ] UV Decoder入力: mapped_features + base(32) + style(512)
- [ ] 出力がlatent32, opacity, scale, rotationの全てを含む

### Phase 4: Rendering
- [ ] Gaussian attributesが正しい範囲 (opacity: sigmoid, scale: exp)
- [ ] Bone matricesが正しくアニメーション
- [ ] Alpha blendingが正しく動作
- [ ] Neural refinerが32ch latentを受け取る

## 次のステップ

1. `image-encoder-corrected.ts`をテスト
2. Inverse texture mappingの完全実装
3. UV/Template decoderの入力検証
4. End-to-endパイプラインの統合テスト
5. 公式モデルweightsとの出力比較

## 参考リンク

- 論文 Fig. 2: Full pipeline diagram
- ubody_gaussian.py lines 116-159: Forward pass
- data_loader.py lines 377-394: Canonical camera params
