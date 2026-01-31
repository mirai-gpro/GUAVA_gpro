# GUAVA TypeScript Pipeline - Complete Specification

## Pipeline Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SOURCE IMAGE (518×518 RGB)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: IMAGE ENCODER                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ dinov2_518.onnx + dinov2_518.onnx.data                                  │   │
│  │   Input:  pixel_values [1, 3, 518, 518]  (normalized RGB)               │   │
│  │   Output: last_hidden_state [1, 1370, 768]  (CLS + 37×37 patches)       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ dino_encoder.onnx                                                       │   │
│  │   Input:  dinov2_features [1, 768, 37, 37]                              │   │
│  │   Output: appearance_features [1, 160, 518, 518]                        │   │
│  │           ├─ channels 0-127:   Template branch (128ch)                  │   │
│  │           └─ channels 128-159: UV branch (32ch) ← ⚠️ OR channels 0-31  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Outputs:                                                                       │
│    • f_global: [768]         (CLS token for global embedding)                   │
│    • f_template: [518,518,128] (Template branch, HWC)                          │
│    • f_uv: [518,518,32]        (UV branch, HWC)                                │
│    • source_rgb: [518,518,3]   (Saved for inverse mapping)                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                     │                                              │
                     │                                              │
        ┌────────────┴────────────┐                    ┌───────────┴───────────┐
        ▼                         ▼                    ▼                       ▼
┌───────────────────┐    ┌───────────────────┐ ┌───────────────────┐  ┌─────────────┐
│ TEMPLATE BRANCH   │    │ UV BRANCH         │ │ source_rgb        │  │ f_global    │
│ f_template (128ch)│    │ f_uv (32ch)       │ │ (3ch)             │  │ (768ch)     │
└───────────────────┘    └───────────────────┘ └───────────────────┘  └─────────────┘
        │                         │                    │                    │
        ▼                         └────────────────────┴────────────────────┘
┌───────────────────────────┐                          │
│ STEP 2: TEMPLATE DECODER  │                          ▼
│ (Projection Sampling)     │          ┌───────────────────────────────────────┐
└───────────────────────────┘          │ STEP 3: INVERSE TEXTURE MAPPING       │
        │                              │ (Image Space → UV Space)               │
        ▼                              │                                        │
┌─────────────────────────────────┐    │   3D World Position ← UV mapping      │
│ template_decoder.onnx           │    │   → Project to Screen                 │
│                                 │    │   → Sample from f_uv (32ch)           │
│ Inputs:                         │    │   → Sample from source_rgb (3ch)      │
│   projection_features [1,N,128] │    │                                        │
│   global_embedding [1,768]      │    │ Output:                               │
│   base_features [1,N,128]       │    │   uv_features_1024 [1024,1024,32]     │
│   view_dirs [1,27]              │    │   uv_rgb_1024 [1024,1024,3]           │
│                                 │    └───────────────────────────────────────┘
│ Outputs:                        │                          │
│   rgb [1,N,32]                  │                          ▼
│   opacity [1,N,1]               │    ┌───────────────────────────────────────┐
│   scale [1,N,3]                 │    │ STEP 4: RESAMPLE (1024→512)           │
│   rotation [1,N,4]              │    │   uv_features [512,512,32] → CHW      │
│   offset [1,N,3]                │    │   uv_rgb [512,512,3] → CHW            │
│   id_embedding_256 [1,256]      │    └───────────────────────────────────────┘
│                                 │                          │
│ N = 10595 (SMPLX vertices)      │                          ▼
└─────────────────────────────────┘    ┌───────────────────────────────────────┐
        │                              │ STEP 5: STYLE MAPPING                  │
        │                              │ uv_style_mapping.onnx                  │
        │                              │                                        │
        │                              │   Input:  global_feature [1, 768]     │
        │                              │   Output: extra_style [1, 512]        │
        │                              └───────────────────────────────────────┘
        │                                                    │
        │                                                    ▼
        │                              ┌───────────────────────────────────────┐
        │                              │ STEP 6: UV STYLEUNET                   │
        │                              │ light_styleunet_fp16.onnx (3.69MB)     │
        │                              │                                        │
        │                              │ Input:                                 │
        │                              │   uv_features [1, 35, 512, 512]       │
        │                              │   (= 32ch UV + 3ch RGB, CHW)           │
        │                              │   extra_style [1, 512]                │
        │                              │                                        │
        │                              │ Output:                                │
        │                              │   styleunet_output [1, 96, 512, 512]  │
        │                              └───────────────────────────────────────┘
        │                                                    │
        │                                                    ▼
        │                              ┌───────────────────────────────────────┐
        │                              │ STEP 7: ADD BASE FEATURE               │
        │                              │ uv_base_feature.bin                    │
        │                              │                                        │
        │                              │   Input: styleunet_output [96ch]      │
        │                              │   Base:  uv_base_feature [32ch]       │
        │                              │   Output: features_128ch [128ch]      │
        │                              │                                        │
        │                              │   96ch + 32ch = 128ch                  │
        │                              └───────────────────────────────────────┘
        │                                                    │
        │                                                    ▼
        │                              ┌───────────────────────────────────────┐
        │                              │ STEP 8: ADD VIEW ENCODING              │
        │                              │                                        │
        │                              │   Input: features_128ch [128ch]       │
        │                              │   View:  view_dirs [27ch] SH encoding │
        │                              │   Output: features_155ch [155ch]      │
        │                              │                                        │
        │                              │   128ch + 27ch = 155ch                 │
        │                              └───────────────────────────────────────┘
        │                                                    │
        │                                                    ▼
        │                              ┌───────────────────────────────────────┐
        │                              │ STEP 9: UV DECODER                     │
        │                              │ uv_point_decoder.onnx                  │
        │                              │                                        │
        │                              │ Input:                                 │
        │                              │   uv_features [1, 155, 512, 512]      │
        │                              │                                        │
        │                              │ Output (per UV pixel → N Gaussians):  │
        │                              │   local_pos [1, 512, 512, 3]          │
        │                              │   opacity [1, 512, 512, 1] (sigmoid)  │
        │                              │   scale [1, 512, 512, 3] (exp)        │
        │                              │   rotation [1, 512, 512, 4] (norm)    │
        │                              │   colors [1, 512, 512, 32]            │
        │                              │                                        │
        │                              │ Valid UV pixels: ~250K Gaussians       │
        │                              └───────────────────────────────────────┘
        │                                                    │
        │                                                    │
        ▼                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                      STEP 10: MERGE UBODY GAUSSIANS                           │
│                                                                               │
│   Template Gaussians (10595)  +  UV Gaussians (~250K)  =  Ubody Gaussians    │
│                                                                               │
│   For each Gaussian:                                                          │
│     • position [3]      (world space)                                         │
│     • scale [3]         (log-scale)                                           │
│     • rotation [4]      (quaternion wxyz)                                     │
│     • opacity [1]       (inverse-sigmoid)                                     │
│     • latent32ch [32]   (appearance features, first 3ch = sigmoid RGB)        │
│                                                                               │
│   ⚠️ sigmoid(latent32ch[0:3]) applied here for RGB                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                      STEP 11: GAUSSIAN SPLATTING                              │
│                      gaussian-rasterizer-webgpu.ts                            │
│                                                                               │
│   EWA Splatting Algorithm:                                                    │
│   1. Compute 3D covariance: Σ_3D = (S*R)^T * (S*R)                           │
│   2. Project to 2D: Σ_2D = T^T * Σ_3D * T                                    │
│   3. Compute conic (inverse covariance) for anisotropic rendering            │
│   4. Alpha compositing with transmittance                                     │
│                                                                               │
│   Input:                                                                      │
│     • positions [N, 3]                                                        │
│     • scales [N, 3]                                                           │
│     • rotations [N, 4]                                                        │
│     • opacities [N]                                                           │
│     • features [N, 32]                                                        │
│     • camera params (view matrix, focal length)                               │
│                                                                               │
│   Output:                                                                     │
│     • coarse_feature_map [32, 512, 512]  (CHW format)                        │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                      STEP 12: NEURAL REFINER                                  │
│                      simpleunet_trained.onnx (38MB)                           │
│                                                                               │
│   Input:                                                                      │
│     • feature_map [1, 32, 512, 512]  (CHW format)                            │
│     • First 3 channels: RGB (already sigmoidized)                             │
│     • Channels 3-31: latent features                                          │
│                                                                               │
│   Output:                                                                     │
│     • rgb_image [1, 3, 512, 512]  (CHW format)                               │
│                                                                               │
│   ⚠️ Post-processing:                                                         │
│     • If raw output range suggests pre-sigmoid: apply sigmoid                 │
│     • Clamp to [0, 1]                                                         │
│     • Convert CHW → HWC for display                                          │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           FINAL OUTPUT (512×512 RGB)                          │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Files Summary

| File | Size | Input | Output | Notes |
|------|------|-------|--------|-------|
| `dinov2_518.onnx` | ~350MB | `[1,3,518,518]` | `[1,1370,768]` | DINOv2 ViT-B/14 |
| `dinov2_518.onnx.data` | ~350MB | (external weights) | - | Separate weights file |
| `dino_encoder.onnx` | ~100MB | `[1,768,37,37]` | `[1,160,518,518]` | Feature upsampler |
| `template_decoder.onnx` | ~50MB | See below | See below | Template Gaussian decoder |
| `uv_style_mapping.onnx` | ~2MB | `[1,768]` | `[1,512]` | Global→Style mapping |
| `light_styleunet_fp16.onnx` | **3.69MB** | `[1,35,H,W]`+`[1,512]` | `[1,96,H,W]` | FP16 quantized |
| `uv_point_decoder.onnx` | ~20MB | `[1,155,H,W]` | Multiple | UV Gaussian decoder |
| `simpleunet_trained.onnx` | **38MB** | `[1,32,512,512]` | `[1,3,512,512]` | Neural refiner |

---

## Binary Data Files

| File | Format | Shape | Description |
|------|--------|-------|-------------|
| `v_template.bin` | Float32 | `[10595, 3]` | SMPLX template vertices |
| `uv_coord.bin` | Float32 | `[10595, 2]` | Per-vertex UV coordinates |
| `vertex_base_feature.bin` | Float32 | `[10595, 128]` | Learned vertex features |
| `uv_base_feature.bin` | Float32 | `[32, 512, 512]` | Learned UV base features |
| `smplx_faces.bin` | Uint32 | `[20908, 3]` | Triangle indices |
| `uv_coords.bin` | Float32 | `[10595, 2]` | UV mapping coordinates |

---

## Detailed Model Specifications

### 1. template_decoder.onnx

**Inputs:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `projection_features` | `[1, 10595, 128]` | float32 | Sampled image features per vertex |
| `global_embedding` | `[1, 768]` | float32 | CLS token from DINOv2 |
| `base_features` | `[1, 10595, 128]` | float32 | Learned vertex features |
| `view_dirs` | `[1, 27]` | float32 | Spherical harmonics view encoding |

**Outputs:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `rgb` | `[1, 10595, 32]` | float32 | Latent color features (pre-sigmoid RGB in [0:3]) |
| `opacity` | `[1, 10595, 1]` | float32 | Opacity (sigmoid applied in Python) |
| `scale` | `[1, 10595, 3]` | float32 | Scale (sigmoid * 0.05 in Python) |
| `rotation` | `[1, 10595, 4]` | float32 | Rotation quaternion (normalized) |
| `offset` | `[1, 10595, 3]` | float32 | Position offset from template |
| `id_embedding_256` | `[1, 256]` | float32 | Identity embedding |

---

### 2. uv_point_decoder.onnx

**Input:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `uv_features` | `[1, 155, H, W]` | float32 | 96 (StyleUNet) + 32 (base) + 27 (view_dirs) |

**Outputs:**
| Name | Shape | Type | Activation | Description |
|------|-------|------|------------|-------------|
| `local_pos` | `[1, H, W, 3]` | float32 | none | Local position offset |
| `opacity` | `[1, H, W, 1]` | float32 | **sigmoid** | Opacity [0,1] |
| `scale` | `[1, H, W, 3]` | float32 | **exp** | Scale (log-space) |
| `rotation` | `[1, H, W, 4]` | float32 | **normalize** | Quaternion |
| `colors` | `[1, H, W, 32]` | float32 | none | Latent features |

---

### 3. simpleunet_trained.onnx (Neural Refiner)

**Input:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input` | `[1, 32, 512, 512]` | float32 | Splatted feature map (CHW) |

**Output:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `output` | `[1, 3, 512, 512]` | float32 | RGB image (CHW, needs clamp [0,1]) |

---

## ⚠️ Known Issues / Mismatches

### 1. DINO Encoder Output Format
**Python:**
```python
self.output_conv = nn.Conv2d(hidden_dims, output_dim, ...)   # 32ch (UV)
self.output_conv_2 = nn.Conv2d(hidden_dims, output_dim_2, ...)  # 128ch (Template)
return {'f_map1': out, 'f_map2': out_2, 'f_global': out_global}
```

**TypeScript:**
```typescript
// Uses first 32ch of 128ch output as UV features (SUBSET!)
if (appearanceChannels === 128) {
  for (let c = 0; c < 32; c++) {
    this.uvFeatures[i * 32 + c] = appearanceData[c * numPixels + i];
  }
}
```

**⚠️ PROBLEM:** TypeScript uses channels 0-31 as subset, Python has SEPARATE output layers!

---

### 2. Inverse Texture Mapping - Missing Visibility Mask
**Python:**
```python
visible_faces, fragments = self.mesh_renderer.render_fragments(...)
uvmap_features = self.convert_pixel_feature_to_uv(..., visible_faces=visible_faces)
```

**TypeScript:** No visibility mask - all UV pixels sampled regardless of occlusion.

---

### 3. Model Compatibility
- `simpleunet_trained.onnx` - Was it trained on the SAME latent32ch distribution?
- `dino_encoder.onnx` - Does it output 160ch (128+32) or just 128ch?

---

## View Direction Encoding

27-dimensional Spherical Harmonics encoding:
```
view_dirs = [
  x, y, z,                           // L0: 3
  xy, yz, xz, x²-y², 3z²-1,         // L1: 5
  ... (higher order terms)           // L2-L4: 19
]
Total: 3 + 5 + 7 + 9 + 3 = 27 dimensions
```

---

## Camera Parameters

**Canonical Camera (from GUAVA paper):**
```
tanfov = 1/24
image_size = 512×512
w2c_cam translation = [0, 0.6, 22]
```

**Focal Length Calculation:**
```typescript
focalY = height / (2 * tan(fov/2))
focalX = focalY * aspect_ratio
```
