#!/usr/bin/env python3
"""
export_complete_template_decoder.py

論文準拠の完全なTemplate Decoderをエクスポート

公式実装 (ubody_gaussian.py) の正確な再現:
1. global_feature_mapping: CLS token [768] → ID embedding [256]
2. Feature concatenation: [projection(128) + base(128) + global(256)] → [512]
3. vertex_gs_decoder: Fused features [512] → Gaussian attributes

このスクリプトは上記の全てを1つのONNXモデルに統合
"""

import torch
import torch.nn as nn
import os
import numpy as np

# ==============================================================================
# Complete Template Decoder (Global Mapping + GS Decoder)
# ==============================================================================
class CompleteTemplateDecoder(nn.Module):
    """
    公式実装の完全な再現
    
    構成:
    1. global_feature_mapping (768 → 256)
    2. vertex_base_feature (学習済み) [N, 128]
    3. vertex_gs_decoder (512 → Gaussian attributes)
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
        # vertex_base_feature は [total_vertices, 128] なので最初のN個を使用
        base_for_batch = self.vertex_base_feature[:N, :]  # [N, 128]
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
    print("🚀 Complete Template Decoder Export (論文準拠)")
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
    print(f"\n[5/5] Building complete template decoder...")
    
    complete_model = CompleteTemplateDecoder(
        global_mapping=global_mapping,
        vertex_base_feature=vertex_base_feature,
        gs_decoder=gs_decoder
    )
    complete_model.eval()
    
    print("   ✅ Complete model assembled")
    
    # ========== Export to ONNX ==========
    print(f"\n📦 Exporting to ONNX...")
    
    N = 100  # Dynamic axis
    dummy_projection = torch.randn(1, N, 128)
    dummy_global = torch.randn(1, 768)
    dummy_view_dirs = torch.randn(1, 27)
    
    with torch.no_grad():
        torch.onnx.export(
            complete_model,
            (dummy_projection, dummy_global, dummy_view_dirs),
            OUTPUT_PATH,
            export_params=True,
            opset_version=14,
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
            # レガシーAPIを使用（新しいexporterでも動作）
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
        print(f"   ✅ Using external data file (recommended for large models)")
    
    # ========== Verification ==========
    print(f"\n🔍 Verification...")
    
    # Test forward pass
    with torch.no_grad():
        test_proj = torch.randn(1, 10, 128)
        test_global = torch.randn(1, 768)
        test_dirs = torch.randn(1, 27)
        
        rgb, opacity, scale, rotation, offset, id_emb = complete_model(test_proj, test_global, test_dirs)
        
        print(f"   Input shapes:")
        print(f"     projection_features: {list(test_proj.shape)}")
        print(f"     global_embedding: {list(test_global.shape)}")
        print(f"     view_dirs: {list(test_dirs.shape)}")
        print(f"   Output shapes:")
        print(f"     rgb: {list(rgb.shape)}")
        print(f"     opacity: {list(opacity.shape)}")
        print(f"     scale: {list(scale.shape)}")
        print(f"     rotation: {list(rotation.shape)}")
        print(f"     offset: {list(offset.shape)}")
        print(f"     id_embedding_256: {list(id_emb.shape)}")
        print(f"   Output ranges:")
        print(f"     rgb: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"     opacity: [{opacity.min():.4f}, {opacity.max():.4f}]")
        print(f"     scale: [{scale.min():.4f}, {scale.max():.4f}]")
        print(f"     rotation: [{rotation.min():.4f}, {rotation.max():.4f}]")
        print(f"     id_embedding_256: [{id_emb.min():.4f}, {id_emb.max():.4f}]")
    
    print("\n" + "=" * 70)
    print("✅ Complete Template Decoder Export Success!")
    print("=" * 70)
    print(f"\n📋 Summary:")
    print(f"   - Global Feature Mapping: ✅ Included (768→256)")
    print(f"   - Vertex Base Features: ✅ Embedded ({num_vertices} vertices)")
    print(f"   - GS Decoder: ✅ Included")
    print(f"\n📝 Next steps:")
    print(f"   1. Copy BOTH files to public/assets/:")
    print(f"      - {OUTPUT_PATH}")
    if os.path.exists(OUTPUT_PATH + ".data"):
        print(f"      - {OUTPUT_PATH}.data")
    print(f"   2. Update TypeScript files:")
    print(f"      - image-encoder-complete.ts → image-encoder.ts")
    print(f"      - template-decoder-complete.ts → template-decoder-onnx.ts")
    print(f"      - gvrm-complete.ts → gvrm.ts")
    print(f"   3. The TypeScript loader will automatically handle .data file")
    print()
    
    return True


if __name__ == "__main__":
    success = export_complete_decoder()
    exit(0 if success else 1)
