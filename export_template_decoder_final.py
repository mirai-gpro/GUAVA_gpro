import torch
import torch.nn as nn
import onnx
import sys
import os

# パス設定
sys.path.append(".") 
from models.modules.net_module.feature_decoder import Vertex_GS_Decoder

def export_decoder():
    print("🚀 Template Decoder (Final V3) のエクスポート...")
    
    WEIGHTS_PATH = "best_160000.pt"
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ {WEIGHTS_PATH} がありません。")
        return

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    # キーの抽出
    new_state_dict = {}
    prefix = "vertex_gs_decoder."
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v

    # in_dim=512 (確定)
    model = Vertex_GS_Decoder(in_dim=512, dir_dim=27, color_out_dim=32)
    
    print("⚖️  Loading state dict...")
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("🎉 Perfect Match! (Strict=True)")
    except Exception as e:
        print("❌ Error:", e)
        return

    model.eval()

    # ★修正箇所: ダミー入力の形★
    N = 100 
    dummy_feat = torch.randn(1, N, 512) # [Batch, Vertices, FeatureDim]
    dummy_dir = torch.randn(1, 27)      # [Batch, ViewDim] ← ここを修正！(Nを削除)

    output_path = "template_decoder.onnx"
    
    print("📦 ONNXエクスポート実行中...")
    torch.onnx.export(
        model,
        (dummy_feat, dummy_dir),
        output_path,
        input_names=['fused_features', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset'],
        opset_version=14,
        dynamic_axes={
            'fused_features': {0: 'batch', 1: 'num_vertices'},
            'view_dirs': {0: 'batch'}, # ViewDirは頂点数に依存しない
            'rgb': {0: 'batch', 1: 'num_vertices'},
            'opacity': {0: 'batch', 1: 'num_vertices'},
            'scale': {0: 'batch', 1: 'num_vertices'},
            'rotation': {0: 'batch', 1: 'num_vertices'},
            'offset': {0: 'batch', 1: 'num_vertices'}
        }
    )
    
    print(f"\n✅ 生成完了: {output_path}")

if __name__ == "__main__":
    export_decoder()