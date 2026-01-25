#!/usr/bin/env python3
"""
export_all_guava_assets.py

best_160000.pt から GUAVA パイプラインに必要な全てのアセットを生成

生成ファイル:
1. base_features.bin - 頂点ごとの学習済み特徴 [N, 128]
2. global_feature_mapping.onnx - CLS token 768→256変換
3. template_decoder.onnx - Template Gaussian生成
"""

import torch
import torch.nn as nn
import onnx
import os
import numpy as np

# ==============================================================================
# 1. Global Feature Mapping (768 → 256)
# ==============================================================================
class GlobalFeatureMappingWrapper(nn.Module):
    """
    global_feature_mapping を単独でエクスポートするためのラッパー
    
    公式実装 (ubody_gaussian.py lines 40-42):
        self.global_feature_mapping = nn.Sequential(
            nn.Linear(768, cfg.global_vertex_dim),      # 768 → 256
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.global_vertex_dim, cfg.global_vertex_dim),  # 256 → 256
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.global_vertex_dim, cfg.global_vertex_dim)   # 256 → 256
        )
    """
    def __init__(self, global_mapping):
        super().__init__()
        self.mapping = global_mapping
    
    def forward(self, cls_token):
        """
        Args:
            cls_token: [B, 768] DINOv2 CLS token
        Returns:
            id_embedding: [B, 256] ID embedding
        """
        return self.mapping(cls_token)


# ==============================================================================
# 2. Template Decoder (修正版)
# ==============================================================================
class Vertex_GS_Decoder_Fixed(nn.Module):
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
# メイン処理
# ==============================================================================
def export_all_assets():
    print("=" * 70)
    print("🚀 GUAVA Asset Exporter")
    print("=" * 70)
    
    CHECKPOINT_PATH = "best_160000.pt"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ {CHECKPOINT_PATH} not found")
        return
    
    print(f"\n[1/4] Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    print(f"✅ Loaded with {len(state_dict)} keys")
    
    # ========== 1. Base Features ==========
    print("\n[2/4] Extracting base features...")
    if 'vertex_base_feature' in state_dict:
        base_feat = state_dict['vertex_base_feature'].float().numpy()
        output_path = "base_features.bin"
        base_feat.tofile(output_path)
        
        num_vertices = base_feat.shape[0]
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ Saved: {output_path}")
        print(f"   Shape: [{num_vertices}, 128]")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Stats: min={base_feat.min():.4f}, max={base_feat.max():.4f}")
    else:
        print("❌ vertex_base_feature not found in checkpoint")
        return
    
    # ========== 2. Global Feature Mapping ==========
    print("\n[3/4] Exporting global_feature_mapping...")
    
    # global_feature_mapping の重みを抽出
    global_mapping_dict = {}
    prefix = "global_feature_mapping."
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = "mapping." + k[len(prefix):]
            global_mapping_dict[new_key] = v
    
    if not global_mapping_dict:
        print("❌ global_feature_mapping not found in checkpoint")
        print("Available keys starting with 'global':")
        for k in state_dict.keys():
            if 'global' in k.lower():
                print(f"  - {k}")
        return
    
    print(f"Found {len(global_mapping_dict)} layers")
    for k in sorted(global_mapping_dict.keys()):
        print(f"  - {k}: {global_mapping_dict[k].shape}")
    
    # モデル構築 (cfg.global_vertex_dim = 256 と仮定)
    global_mapping = nn.Sequential(
        nn.Linear(768, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 256)
    )
    
    wrapper = GlobalFeatureMappingWrapper(global_mapping)
    
    # 重みロード
    try:
        wrapper.load_state_dict(global_mapping_dict, strict=True)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"⚠️ Load error: {e}")
        wrapper.load_state_dict(global_mapping_dict, strict=False)
    
    wrapper.eval()
    
    # ONNX エクスポート
    dummy_cls = torch.randn(1, 768)
    output_path = "global_feature_mapping.onnx"
    
    torch.onnx.export(
        wrapper,
        dummy_cls,
        output_path,
        input_names=['cls_token'],
        output_names=['id_embedding'],
        dynamic_axes={'cls_token': {0: 'batch'}},
        opset_version=14
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    
    # 検証
    with torch.no_grad():
        test_out = wrapper(dummy_cls)
        print(f"   Test output shape: {test_out.shape}")
        print(f"   Test output stats: min={test_out.min():.4f}, max={test_out.max():.4f}")
    
    # ========== 3. Template Decoder ==========
    print("\n[4/4] Exporting template_decoder...")
    
    decoder_dict = {}
    prefix = "vertex_gs_decoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            decoder_dict[k[len(prefix):]] = v
    
    model = Vertex_GS_Decoder_Fixed(in_dim=512, dir_dim=27, color_out_dim=32)
    
    try:
        model.load_state_dict(decoder_dict, strict=True)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"⚠️ Load warning: {e}")
        model.load_state_dict(decoder_dict, strict=False)
    
    model.eval()
    
    N = 100
    dummy_feat = torch.randn(1, N, 512)
    dummy_dir = torch.randn(1, 27)
    output_path = "template_decoder.onnx"
    
    torch.onnx.export(
        model,
        (dummy_feat, dummy_dir),
        output_path,
        input_names=['fused_features', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset'],
        opset_version=14,
        dynamic_axes={
            'fused_features': {1: 'num_vertices'},
            'rgb': {1: 'num_vertices'},
            'opacity': {1: 'num_vertices'},
            'scale': {1: 'num_vertices'},
            'rotation': {1: 'num_vertices'},
            'offset': {1: 'num_vertices'}
        }
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    
    # ========== 完了 ==========
    print("\n" + "=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. base_features.bin              - Vertex base features")
    print("  2. global_feature_mapping.onnx    - CLS token → ID embedding")
    print("  3. template_decoder.onnx          - Template Gaussian decoder")
    print("\nNext steps:")
    print("  1. Copy all files to public/assets/")
    print("  2. Update image-encoder.ts to use global_feature_mapping.onnx")
    print("  3. Test the pipeline")
    print()


if __name__ == "__main__":
    export_all_assets()
