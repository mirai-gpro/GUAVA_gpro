import torch
import torch.nn as nn
import onnx
import sys
import os

# ==========================================
# 1. クラス定義 (Offsetなし・修正版)
# ==========================================
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
        # Offsetはゼロ固定
        offset = torch.zeros(B, N, 3, device=features.device)
        
        return rgb, opacity, scale, rotation, offset

# ==========================================
# 2. パッキング実行 (Opset 18 - Native)
# ==========================================
def pack_and_export_v18():
    print("🚀 Packing Template Decoder (Opset 18 - Native)...")
    
    weights_path = "best_160000.pt"
    if not os.path.exists(weights_path):
        print(f"❌ '{weights_path}' がありません。")
        return

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
        print("✅ Weights loaded into Fixed Model.")
    except Exception as e:
        print(f"⚠️ Load warning: {e}")

    model.eval()

    # ダミー入力
    N = 100
    dummy_feat = torch.randn(1, N, 512)
    dummy_dir = torch.randn(1, 27)

    output_path = "template_decoder.onnx"

    # ★修正: opset_version=18 (環境のネイティブバージョンを使用)
    torch.onnx.export(
        model,
        (dummy_feat, dummy_dir),
        output_path,
        input_names=['fused_features', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset'],
        opset_version=18, # Native output without conversion
        export_params=True,
        do_constant_folding=True,
        dynamic_axes={
            'fused_features': {1: 'num_vertices'},
            'rgb': {1: 'num_vertices'},
            'opacity': {1: 'num_vertices'},
            'scale': {1: 'num_vertices'},
            'rotation': {1: 'num_vertices'},
            'offset': {1: 'num_vertices'}
        }
    )
    
    print("📦 Verifying file size...")
    
    # ファイルサイズチェック
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"📊 Generated size: {size_mb:.2f} MB")
    
    if size_mb < 0.1:
        print("❌ 失敗: まだファイルが小さすぎます。")
    else:
        # ONNXライブラリでロードして再保存 (念のためのパッキング処理)
        onnx_model = onnx.load(output_path)
        onnx.save(onnx_model, output_path)
        print(f"✅ SUCCESS! Packed model saved to: {output_path}")
        print("これを assets に上書きしてください。")

if __name__ == "__main__":
    pack_and_export_v18()