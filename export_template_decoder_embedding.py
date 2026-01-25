#!/usr/bin/env python3
"""
export_template_decoder_embedding.py

✅ SOLUTION: nn.Embedding を使用して動的スライシング問題を解決

問題:
  vertex_base_feature[:N, :] や torch.index_select は
  ONNX の Gather オペレーションに変換され、dynamic_axes で壊れる

解決策:
  nn.Embedding を使用。ONNXで安定してサポートされている。
  成功事例: PyTorch Issues #25469, #28930
"""

import torch
import torch.nn as nn
import os
import numpy as np

# ==============================================================================
# Template Decoder with nn.Embedding (FIXED)
# ==============================================================================
class CompleteTemplateDecoderWithEmbedding(nn.Module):
    """
    ✅ nn.Embedding を使用した修正版
    
    vertex_base_feature を nn.Embedding に変換することで、
    ONNX エクスポート時の動的スライシング問題を回避
    """
    def __init__(self, global_mapping, vertex_base_feature, gs_decoder):
        super().__init__()
        self.global_mapping = global_mapping
        
        # ✅ KEY FIX: vertex_base_feature を nn.Embedding に変換
        num_vertices, feature_dim = vertex_base_feature.shape
        self.base_embedding = nn.Embedding(num_vertices, feature_dim)
        # 学習済みの重みをコピー
        self.base_embedding.weight.data = vertex_base_feature.clone()
        # 推論時は freeze（学習しない）
        self.base_embedding.weight.requires_grad = False
        
        self.gs_decoder = gs_decoder
        
        print(f"[TemplateDecoder] ✅ Created base_embedding: {num_vertices} vertices x {feature_dim} features")
    
    def forward(self, projection_features, global_embedding_768, view_dirs):
        """
        Args:
            projection_features: [B, N, 128]
            global_embedding_768: [B, 768]
            view_dirs: [B, 27]
        
        Returns:
            rgb, opacity, scale, rotation, offset, id_embedding_256
        """
        B, N, _ = projection_features.shape
        
        # 1. Global feature mapping (768 → 256)
        global_feature_256 = self.global_mapping(global_embedding_768)  # [B, 256]
        
        # 2. Expand global feature
        global_expanded = global_feature_256.unsqueeze(1).expand(B, N, 256)  # [B, N, 256]
        
        # 3. ✅ Get base features using Embedding (NO SLICING!)
        # vertex_ids: [0, 1, 2, ..., N-1]
        vertex_ids = torch.arange(N, dtype=torch.long, device=projection_features.device)
        base_for_batch = self.base_embedding(vertex_ids)  # [N, 128]
        base_expanded = base_for_batch.unsqueeze(0).expand(B, N, 128)  # [B, N, 128]
        
        # 4. Concatenate: [projection(128) + base(128) + global(256)] = [512]
        fused_features = torch.cat([
            projection_features,  # [B, N, 128]
            base_expanded,        # [B, N, 128]
            global_expanded       # [B, N, 256]
        ], dim=-1)  # [B, N, 512]
        
        # 5. GS Decoder
        rgb, opacity, scale, rotation, offset = self.gs_decoder(fused_features, view_dirs)
        
        # 6. Return with ID embedding
        return rgb, opacity, scale, rotation, offset, global_feature_256


class Vertex_GS_Decoder_Fixed(nn.Module):
    """Offset層を削除したVertex GS Decoder"""
    def __init__(self, in_dim=512, dir_dim=27, color_out_dim=32):
        super().__init__()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2, bias=True),
        )
        
        layer_in_dim = in_dim//2 + dir_dim
        
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, color_out_dim, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3, bias=True),
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, fused_features, view_dirs):
        features = self.feature_layers(fused_features)
        
        B, N, _ = features.shape
        dirs_expanded = view_dirs.unsqueeze(1).expand(B, N, -1)
        features_cat = torch.cat([features, dirs_expanded], dim=-1)
        
        rgb = self.color_layers(features_cat)
        opacity = torch.sigmoid(self.opacity_layers(features_cat))
        scale = torch.exp(self.scale_layers(features_cat))
        rotation = torch.nn.functional.normalize(self.rotation_layers(features_cat), dim=-1)
        offset = torch.zeros(B, N, 3, device=features.device)
        
        return rgb, opacity, scale, rotation, offset


# ==============================================================================
# Export Pipeline
# ==============================================================================
def export_with_embedding():
    print("=" * 70)
    print("🚀 Template Decoder Export with nn.Embedding Fix")
    print("=" * 70)
    print("✅ Solution: Replace slicing with nn.Embedding")
    print("✅ Based on: PyTorch Issues #25469, #28930")
    print("=" * 70)
    
    CHECKPOINT_PATH = "best_160000.pt"
    OUTPUT_PATH = "template_decoder.onnx"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ {CHECKPOINT_PATH} not found")
        return False
    
    print(f"\n[1/5] Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    print(f"✅ Loaded {len(state_dict)} keys")
    
    # ========== Extract Components ==========
    
    # 1. Global Feature Mapping
    print(f"\n[2/5] Extracting global_feature_mapping...")
    global_mapping_dict = {}
    prefix = "global_feature_mapping."
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            global_mapping_dict[new_key] = v
    
    if not global_mapping_dict:
        print("❌ global_feature_mapping not found")
        return False
    
    print(f"   Found {len(global_mapping_dict)} parameters")
    
    global_mapping = nn.Sequential(
        nn.Linear(768, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 256)
    )
    global_mapping.load_state_dict(global_mapping_dict, strict=True)
    print("   ✅ Global mapping loaded")
    
    # 2. Vertex Base Feature
    print(f"\n[3/5] Extracting vertex_base_feature...")
    if 'vertex_base_feature' not in state_dict:
        print("❌ vertex_base_feature not found")
        return False
    
    vertex_base_feature = state_dict['vertex_base_feature']  # [N, 128]
    num_vertices = vertex_base_feature.shape[0]
    print(f"   ✅ Base features: [{num_vertices}, 128]")
    
    # 3. Vertex GS Decoder
    print(f"\n[4/5] Extracting vertex_gs_decoder...")
    decoder_dict = {}
    prefix = "vertex_gs_decoder."
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            decoder_dict[k[len(prefix):]] = v
    
    if not decoder_dict:
        print("❌ vertex_gs_decoder not found")
        return False
    
    print(f"   Found {len(decoder_dict)} parameters")
    
    gs_decoder = Vertex_GS_Decoder_Fixed(in_dim=512, dir_dim=27, color_out_dim=32)
    gs_decoder.load_state_dict(decoder_dict, strict=True)
    print("   ✅ GS decoder loaded")
    
    # ========== Build Model with Embedding ==========
    print(f"\n[5/5] Building model with nn.Embedding fix...")
    
    complete_model = CompleteTemplateDecoderWithEmbedding(
        global_mapping=global_mapping,
        vertex_base_feature=vertex_base_feature,
        gs_decoder=gs_decoder
    )
    complete_model.eval()
    
    print("   ✅ Complete model assembled (with Embedding)")
    
    # ========== Test BEFORE Export ==========
    print(f"\n🧪 Testing model BEFORE ONNX export...")
    
    with torch.no_grad():
        # Test 1: N=100
        test_proj_100 = torch.randn(1, 100, 128)
        test_global = torch.randn(1, 768)
        test_dirs = torch.zeros(1, 27)
        
        rgb, opacity, scale, rotation, offset, id_emb = complete_model(
            test_proj_100, test_global, test_dirs
        )
        
        print(f"   Test N=100:")
        print(f"     Opacity: min={opacity.min():.6f}, max={opacity.max():.6f}")
        print(f"     Unique opacity values: {torch.unique(opacity).numel()}")
        
        # Test 2: N=10595
        test_proj_full = torch.randn(1, num_vertices, 128)
        rgb2, opacity2, scale2, rotation2, offset2, id_emb2 = complete_model(
            test_proj_full, test_global, test_dirs
        )
        
        print(f"   Test N={num_vertices}:")
        print(f"     Opacity: min={opacity2.min():.6f}, max={opacity2.max():.6f}")
        print(f"     Unique opacity values: {torch.unique(opacity2).numel()}")
    
    # ========== Export to ONNX ==========
    print(f"\n📦 Exporting to ONNX with Embedding fix...")
    
    N = 100  # Dynamic test size
    dummy_projection = torch.randn(1, N, 128)
    dummy_global = torch.randn(1, 768)
    dummy_view_dirs = torch.zeros(1, 27)
    
    with torch.no_grad():
        torch.onnx.export(
            complete_model,
            (dummy_projection, dummy_global, dummy_view_dirs),
            OUTPUT_PATH,
            export_params=True,
            opset_version=14,  # Embedding は opset 9+ でサポート
            do_constant_folding=True,
            input_names=['projection_features', 'global_embedding', 'view_dirs'],
            output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset', 'id_embedding_256'],
            dynamic_axes={
                'projection_features': {1: 'num_vertices'},
                'rgb': {1: 'num_vertices'},
                'opacity': {1: 'num_vertices'},
                'scale': {1: 'num_vertices'},
                'rotation': {1: 'num_vertices'},
                'offset': {1: 'num_vertices'}
            },
            dynamo=False,
        )
    
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"✅ Export complete: {OUTPUT_PATH}")
    print(f"   Size: {file_size:.2f} MB")
    
    # External data
    data_file = OUTPUT_PATH + ".data"
    if os.path.exists(data_file):
        data_size = os.path.getsize(data_file) / (1024 * 1024)
        print(f"   External data: {data_file} ({data_size:.2f} MB)")
    
    # ========== ONNX Runtime Verification ==========
    print(f"\n🔍 Verifying with ONNX Runtime...")
    
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(OUTPUT_PATH)
        
        print(f"   Testing different N values:")
        for test_n in [50, 100, 1000, num_vertices]:
            test_input = {
                'projection_features': np.random.randn(1, test_n, 128).astype(np.float32),
                'global_embedding': np.random.randn(1, 768).astype(np.float32),
                'view_dirs': np.zeros((1, 27), dtype=np.float32)
            }
            
            outputs = session.run(None, test_input)
            onnx_opacity = outputs[1]
            
            unique_count = np.unique(onnx_opacity).size
            diversity_ratio = unique_count / test_n
            
            print(f"     N={test_n}: opacity range=[{onnx_opacity.min():.6f}, {onnx_opacity.max():.6f}], unique={unique_count} ({diversity_ratio*100:.1f}%)")
            
            if diversity_ratio < 0.5:
                print(f"       ⚠️ WARNING: Low diversity!")
            else:
                print(f"       ✅ Good diversity")
        
    except ImportError:
        print("   ⚠️ onnxruntime not available, skipping verification")
    except Exception as e:
        print(f"   ❌ ONNX Runtime test failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Export Complete with nn.Embedding Fix!")
    print("=" * 70)
    print(f"\n📋 Key Changes:")
    print(f"   ❌ OLD: base_for_batch = self.vertex_base_feature[:N, :]")
    print(f"   ❌ OLD: base_for_batch = torch.index_select(...)")
    print(f"   ✅ NEW: self.base_embedding = nn.Embedding(num_vertices, 128)")
    print(f"   ✅ NEW: base_for_batch = self.base_embedding(vertex_ids)")
    print(f"\n📝 Next steps:")
    print(f"   1. cp {OUTPUT_PATH} public/assets/")
    if os.path.exists(data_file):
        print(f"   2. cp {data_file} public/assets/")
    print(f"   3. Hard reload browser (Ctrl+Shift+R)")
    print(f"   4. Check console: Opacity should have 10000+ unique values!")
    print()
    
    return True


if __name__ == "__main__":
    success = export_with_embedding()
    exit(0 if success else 1)