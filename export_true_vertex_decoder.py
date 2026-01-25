import torch
import torch.nn as nn
import onnx
import os
import numpy as np

# ==============================================================================
# 1. クラス定義 (Offset層を削除した修正版)
# ==============================================================================
class Vertex_GS_Decoder_Fixed(nn.Module):
    def __init__(self, in_dim=512, dir_dim=27, color_out_dim=32):
        super().__init__()
        
        # 特徴量抽出 (MLP)
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
        
        # 各属性のヘッド
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
        
        # ★削除: offset_layers は重みファイルに存在しないため削除
        # self.offset_layers = ... 

    def forward(self, fused_features, view_dirs):
        # 1. 特徴量変換
        features = self.feature_layers(fused_features)
        
        # 2. ViewDirの拡張と結合
        B, N, _ = features.shape
        dirs_expanded = view_dirs.unsqueeze(1).expand(B, N, -1)
        features_cat = torch.cat([features, dirs_expanded], dim=-1)
        
        # 3. 各属性の予測
        rgb = self.color_layers(features_cat)
        
        # 安全な活性化関数
        opacity = torch.sigmoid(self.opacity_layers(features_cat))
        scale = torch.exp(self.scale_layers(features_cat))
        rotation = torch.nn.functional.normalize(self.rotation_layers(features_cat), dim=-1)
        
        # ★修正: Offsetはゼロベクトルを出力
        # モデルが学習していないため、変に予測させず「動かない」とするのが正解
        offset = torch.zeros(B, N, 3, device=features.device)
        
        return rgb, opacity, scale, rotation, offset

# ==============================================================================
# 2. エクスポート実行
# ==============================================================================
def export_pipeline():
    print("🚀 Starting Corrected Export Pipeline (No Offset)...")
    
    weights_path = "best_160000.pt"
    if not os.path.exists(weights_path):
        print(f"❌ '{weights_path}' がありません。")
        return

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    # Base Features の再保存 (念のため)
    if 'vertex_base_feature' in state_dict:
        base_feat = state_dict['vertex_base_feature'].float().numpy()
        base_feat.tofile("base_features.bin")
        print(f"✅ Base Features saved ({base_feat.shape})")

    # Decoder重みの抽出
    decoder_dict = {}
    prefix = "vertex_gs_decoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            decoder_dict[k[len(prefix):]] = v

    # モデル構築
    model = Vertex_GS_Decoder_Fixed(in_dim=512, dir_dim=27, color_out_dim=32)
    
    # ロード (今度はStrict=Trueで通るはず！)
    try:
        model.load_state_dict(decoder_dict, strict=True)
        print("🎉 Weights loaded PERFECTLY (Strict Match)!")
    except Exception as e:
        print(f"⚠️ Load warning: {e}")
        # Offsetがないのでエラーは出ないはずだが、念のため
        model.load_state_dict(decoder_dict, strict=False)

    model.eval()

    # エクスポート
    print("📦 Exporting corrected ONNX...")
    N = 100
    dummy_feat = torch.randn(1, N, 512)
    dummy_dir = torch.randn(1, 27)

    output_file = "template_decoder.onnx"

    torch.onnx.export(
        model,
        (dummy_feat, dummy_dir),
        output_file,
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
    
    print(f"✅ Export Complete: {output_file}")
    print("このファイルを assets に上書きしてください。これでNaN地獄から解放されます。")

if __name__ == "__main__":
    export_pipeline()