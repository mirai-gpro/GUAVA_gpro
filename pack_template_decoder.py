import torch
import torch.nn as nn
import onnx
import sys
import os

# パス設定
sys.path.append(".") 
from models.modules.net_module.feature_decoder import Vertex_GS_Decoder

def pack_decoder():
    print("🚀 Template Decoder (Packing Mode) の実行...")
    
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

    # モデル定義 (in_dim=512)
    model = Vertex_GS_Decoder(in_dim=512, dir_dim=27, color_out_dim=32)
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # ダミー入力
    N = 100 
    dummy_feat = torch.randn(1, N, 512)
    dummy_dir = torch.randn(1, 27)

    output_path = "template_decoder_packed.onnx"
    
    print("📦 ONNXエクスポート中...")
    
    # torch.onnx.export で直接出力
    # opset_version=12 にすることで互換性を高め、余計な変換エラーを防ぎます
    torch.onnx.export(
        model,
        (dummy_feat, dummy_dir),
        output_path,
        input_names=['fused_features', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset'],
        opset_version=12, # 14 -> 12 に変更 (安定性重視)
        dynamic_axes={
            'fused_features': {1: 'num_vertices'},
            'rgb': {1: 'num_vertices'},
            'opacity': {1: 'num_vertices'},
            'scale': {1: 'num_vertices'},
            'rotation': {1: 'num_vertices'},
            'offset': {1: 'num_vertices'}
        }
    )
    
    print("📦 ONNXの再検証とパッキング確認...")
    
    # ONNXライブラリを使って「外部データなし」で上書き保存
    onnx_model = onnx.load(output_path)
    
    # チェック: モデルが壊れていないか
    try:
        onnx.checker.check_model(onnx_model)
        print("✅ ONNXフォーマットチェック合格")
    except Exception as e:
        print("⚠️ ONNXチェック警告:", e)

    # 強制的に1ファイルに保存 (2GB以下ならこれがデフォルトですが念のため)
    final_path = "template_decoder.onnx" # ファイル名を戻す
    onnx.save(onnx_model, final_path)
    
    print(f"\n🎉 完了: {final_path}")
    print("このファイルを assets フォルダに上書きしてください。")
    print("(.data ファイルがもしあっても、もう不要です)")

if __name__ == "__main__":
    pack_decoder()