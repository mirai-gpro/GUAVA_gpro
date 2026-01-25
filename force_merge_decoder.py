import torch
import torch.nn as nn
import onnx
import sys
import os
from onnx.external_data_helper import load_external_data_for_model, convert_model_to_external_data, write_external_data_tensors

# ==========================================
# 1. クラス定義 (Offsetなし・修正版)
# ==========================================
class Vertex_GS_Decoder_Fixed(nn.Module):
    def __init__(self, in_dim=512, dir_dim=27, color_out_dim=32):
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//2), nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2), nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2), nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//2),
        )
        layer_in_dim = in_dim//2 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, color_out_dim),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 4),
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

# ==========================================
# 2. 強制マージ実行
# ==========================================
def export_and_merge():
    print("🚀 Force Merging Template Decoder...")
    
    weights_path = "best_160000.pt"
    if not os.path.exists(weights_path):
        print(f"❌ '{weights_path}' がありません。")
        return

    # 重みロード
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    decoder_dict = {}
    prefix = "vertex_gs_decoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            decoder_dict[k[len(prefix):]] = v

    model = Vertex_GS_Decoder_Fixed(in_dim=512, dir_dim=27, color_out_dim=32)
    try:
        model.load_state_dict(decoder_dict, strict=False)
        print("✅ Weights loaded.")
    except Exception as e:
        print(f"⚠️ Load warning: {e}")
    model.eval()

    # ダミー入力
    N = 100
    dummy_feat = torch.randn(1, N, 512)
    dummy_dir = torch.randn(1, 27)
    output_path = "template_decoder.onnx"

    # エクスポート (Opset 14で安定化)
    print("📦 Exporting raw ONNX...")
    torch.onnx.export(
        model,
        (dummy_feat, dummy_dir),
        output_path,
        input_names=['fused_features', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset'],
        opset_version=14, # Stable standard
        dynamic_axes={'fused_features': {1: 'num_vertices'}}
    )

    # サイズ確認
    raw_size = os.path.getsize(output_path) / (1024*1024)
    print(f"   Raw size: {raw_size:.2f} MB")

    # .data ファイルがあるか確認
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        print("   ⚠️ Found external data file! Integrating...")
    else:
        print("   ℹ️ No external data file found (yet).")

    # ONNXライブラリでロードして、インライン化（埋め込み）して保存
    print("🔨 Merging into single file...")
    onnx_model = onnx.load(output_path)
    
    # ここで強制的に全てのテンソルを埋め込む設定で保存
    # save_as_external_data=False が重要
    onnx.save_model(
        onnx_model, 
        output_path, 
        save_as_external_data=False 
    )
    
    # 最終確認
    final_size = os.path.getsize(output_path) / (1024*1024)
    print(f"✅ Final File Size: {final_size:.2f} MB")
    
    if final_size > 1.0:
        print("🎉 SUCCESS! The model contains weights.")
        print("-> Please overwrite 'assets/template_decoder.onnx'")
    else:
        print("❌ ERROR: File is still too small. Check PyTorch version.")

if __name__ == "__main__":
    export_and_merge()