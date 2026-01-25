import torch
import os
import numpy as np

def extract_assets():
    print("🚀 GUAVA Assets Extraction...")
    
    WEIGHTS_PATH = "best_160000.pt"
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ {WEIGHTS_PATH} がありません。")
        return

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    # 1. Base Features (学習済みの頂点特徴量) の抽出
    # キー名: vertex_base_feature
    if 'vertex_base_feature' in state_dict:
        base_feat = state_dict['vertex_base_feature'].float().numpy()
        print(f"✅ vertex_base_feature found: {base_feat.shape}")
        
        # バイナリとして保存 (assets/base_features.bin)
        output_path = "base_features.bin"
        base_feat.tofile(output_path)
        print(f"📦 Saved to {output_path} ({len(base_feat.tobytes())} bytes)")
    else:
        print("❌ vertex_base_feature が見つかりません！")

if __name__ == "__main__":
    extract_assets()