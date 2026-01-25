#!/usr/bin/env python3
"""
export_complete_template_decoder_FIXED.py

⚠️ CRITICAL FIX: vertex_base_feature スライシング問題を解決

問題:
  base_for_batch = self.vertex_base_feature[:N, :]
  
  → ONNXエクスポート時、この動的スライシングが正しく変換されない
  → 結果: 全ての出力が同じ値になる

解決策:
  torch.index_select() を使用して明示的にインデックス指定
"""

import torch
import torch.nn as nn
import os
import numpy as np

# ==============================================================================
# Complete Template Decoder (FIXED VERSION)
# ==============================================================================
class CompleteTemplateDecoderFixed(nn.Module):
    """
    修正版: vertex_base_feature のスライシング問題を解決
    """
    def __init__(self, global_mapping, vertex_base_feature, gs_decoder):
        super().__init__()
        self.global_mapping = global_mapping
        # vertex_base_featureをモデルパラメータとして登録
        self.register_buffer('vertex_base_feature', vertex_base_feature)
        self.gs_decoder = gs_decoder
    
    def forward(self, projection_features, global_embedding_768, view_dirs):
        """
        Args:
            projection_features: [B, N, 128] - Image encoderからの投影特徴
            global_embedding_768: [B, 768] - DINOv2 CLS token
            view_dirs: [B, 27] - Harmonic encoded view direction
        
        Returns:
            rgb: [B, N, 32]
            opacity: [B, N, 1] (sigmoid済み)
            scale: [B, N, 3] (exp済み)
            rotation: [B, N, 4] (正規化済み)
            offset: [B, N, 3] (ゼロ)
            id_embedding_256: [B, 256] ← Neural Refiner用
        """
        B, N, _ = projection_features.shape
        
        # 1. Global feature mapping (768 → 256)
        global_feature_256 = self.global_mapping(global_embedding_768)  # [B, 256]
        
        # 2. Expand global feature to all vertices
        global_expanded = global_feature_256.unsqueeze(1).expand(B, N, 256)  # [B, N, 256]
        
        # 3. Get base features for N vertices
        # ⚠️ FIXED: スライシング [:N, :] の代わりに torch.index_select を使用
        # 理由: ONNXエクスポート時に動的スライシングが壊れる問題を回避
        
        # インデックステンソルを作成
        indices = torch.arange(N, dtype=torch.long, device=projection_features.device)
        
        # index_selectでN個の頂点を選択
        base_for_batch = torch.index_select(self.vertex_base_feature, 0, indices)  # [N, 128]
        
        # バッチ次元を追加して拡張
        base_expanded = base_for_batch.unsqueeze(0).expand(B, N, 128)  # [B, N, 128]
        
        # 4. Concatenate: [projection(128) + base(128) + global(256)] = [512]
        fused_features = torch.cat([
            projection_features,  # [B, N, 128]
            base_expanded,        # [B, N, 128]
            global_expanded       # [B, N, 256]
        ], dim=-1)  # [B, N, 512]
        
        # 5. GS Decoder
        rgb, opacity, scale, rotation, offset = self.gs_decoder(fused_features, view_dirs)
        
        # 6. Return with ID embedding for Neural Refiner
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
def export_complete_decoder():
    print("=" * 70)
    print("🚀 Complete Template Decoder Export (FIXED)")
    print("=" * 70)
    print("⚠️  FIX: vertex_base_feature slicing issue resolved")
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
    
    # ========== Build Complete Model ==========
    print(f"\n[5/5] Building FIXED complete template decoder...")
    
    complete_model = CompleteTemplateDecoderFixed(
        global_mapping=global_mapping,
        vertex_base_feature=vertex_base_feature,
        gs_decoder=gs_decoder
    )
    complete_model.eval()
    
    print("   ✅ Complete model assembled (with index_select fix)")
    
    # ========== Test BEFORE Export ==========
    print(f"\n🧪 Testing model BEFORE ONNX export...")
    
    with torch.no_grad():
        # Test 1: N=100
        test_proj_100 = torch.randn(1, 100, 128)
        test_global = torch.randn(1, 768)
        test_dirs = torch.zeros(1, 27)
        
        rgb, opacity, scale, rotation, offset, id_emb = complete_model(test_proj_100, test_global, test_dirs)
        
        print(f"   Test N=100:")
        print(f"     Opacity: min={opacity.min():.6f}, max={opacity.max():.6f}")
        print(f"     Unique opacity values: {torch.unique(opacity).numel()}")
        
        # Test 2: N=10595 (full)
        test_proj_full = torch.randn(1, num_vertices, 128)
        rgb2, opacity2, scale2, rotation2, offset2, id_emb2 = complete_model(test_proj_full, test_global, test_dirs)
        
        print(f"   Test N={num_vertices}:")
        print(f"     Opacity: min={opacity2.min():.6f}, max={opacity2.max():.6f}")
        print(f"     Unique opacity values: {torch.unique(opacity2).numel()}")
    
    # ========== Export to ONNX ==========
    print(f"\n📦 Exporting to ONNX with FIXED model...")
    
    N = 100  # Dynamic axis test size
    dummy_projection = torch.randn(1, N, 128)
    dummy_global = torch.randn(1, 768)
    dummy_view_dirs = torch.zeros(1, 27)
    
    with torch.no_grad():
        torch.onnx.export(
            complete_model,
            (dummy_projection, dummy_global, dummy_view_dirs),
            OUTPUT_PATH,
            export_params=True,
            opset_version=17,  # Use newer opset for better support
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
    
    # .dataファイルの確認
    data_file = OUTPUT_PATH + ".data"
    if os.path.exists(data_file):
        data_size = os.path.getsize(data_file) / (1024 * 1024)
        print(f"   External data: {data_file} ({data_size:.2f} MB)")
    
    # ========== ONNX Runtime Verification ==========
    print(f"\n🔍 Verifying ONNX model with ONNX Runtime...")
    
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(OUTPUT_PATH)
        
        # Test with different N values
        for test_n in [50, 100, 1000, num_vertices]:
            test_input = {
                'projection_features': np.random.randn(1, test_n, 128).astype(np.float32),
                'global_embedding': np.random.randn(1, 768).astype(np.float32),
                'view_dirs': np.zeros((1, 27), dtype=np.float32)
            }
            
            outputs = session.run(None, test_input)
            onnx_opacity = outputs[1]
            
            unique_count = np.unique(onnx_opacity).size
            print(f"   ONNX Test N={test_n}:")
            print(f"     Opacity: min={onnx_opacity.min():.6f}, max={onnx_opacity.max():.6f}")
            print(f"     Unique values: {unique_count}")
            
            if unique_count < test_n * 0.5:  # Less than 50% unique
                print(f"     ⚠️ WARNING: Too few unique values! Possible slicing issue.")
            else:
                print(f"     ✅ Good diversity")
        
    except ImportError:
        print("   ⚠️ onnxruntime not available, skipping ONNX verification")
    
    print("\n" + "=" * 70)
    print("✅ FIXED Template Decoder Export Success!")
    print("=" * 70)
    print(f"\n📋 Changes:")
    print(f"   ❌ OLD: base_for_batch = self.vertex_base_feature[:N, :]")
    print(f"   ✅ NEW: base_for_batch = torch.index_select(self.vertex_base_feature, 0, indices)")
    print(f"\n📝 Next steps:")
    print(f"   1. Copy to public/assets/:")
    print(f"      cp {OUTPUT_PATH} public/assets/")
    if os.path.exists(data_file):
        print(f"      cp {data_file} public/assets/")
    print(f"   2. Reload the page and check browser console")
    print(f"   3. You should see diverse opacity values now!")
    print()
    
    return True


if __name__ == "__main__":
    success = export_complete_decoder()
    exit(0 if success else 1)