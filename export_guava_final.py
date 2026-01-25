import torch
import torch.nn as nn
import onnx
import sys
import os

# パス設定
sys.path.append(".") 
# もし modules フォルダが認識されない場合は絶対パスを入れるなど調整してください
from models.modules.net_module.styleunet import StyleUNet

# ==========================================
# 1. 完全体モデルの定義 (パラメータ修正版)
# ==========================================
class GuavaRefinerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ★エラーログから判明した「正解の数値」を設定★
        self.refiner = StyleUNet(
            in_size=512, 
            out_size=512, 
            in_dim=35,           # 修正: 32 -> 35
            out_dim=96,          # 修正: 3 -> 96
            num_style_feat=512,
            extra_style_dim=512, # 追加: style_fuseレイヤーを作成するために必須
            activation=False     # ubody_gaussian.pyの設定に合わせる
        )

    def forward(self, feature_map, id_embedding):
        # forwardメソッドも styleunet.py の定義に合わせて修正が必要
        # 多くのStyleUNet実装では forward(x, style_code) ですが、
        # extra_style_dimがある場合、forward(x, style_code, extra_style) となる可能性があります。
        
        # ここでは、id_embedding を style_code として渡します。
        # ※もしエラーが出る場合は引数の順番や個数を調整します。
        
        # 入力データの調整: feature_mapは32chですが、モデルは35chを求めています。
        # TypeScript側で結合するのが正しいですが、ONNXの入力としては
        # 「結合済みの35ch」を受け取るように定義します。
        
        return self.refiner(feature_map, id_embedding)

# ==========================================
# 2. エクスポート処理
# ==========================================
def export():
    print("🚀 GUAVA Refiner (Corrected Config) のエクスポート...")
    
    WEIGHTS_PATH = "best_160000.pt"
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ {WEIGHTS_PATH} がありません。")
        return

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    # キーの書き換え
    new_state_dict = {}
    prefix = "uv_feature_decoder."
    
    print(f"🔑 '{prefix}' から 'refiner.' へキーを変換中...")
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            # uv_feature_decoder.xxx -> refiner.xxx
            new_key = "refiner." + k[len(prefix):]
            new_state_dict[new_key] = v

    # モデル構築
    model = GuavaRefinerWrapper()
    
    # ロード試行
    print("⚖️  Loading state dict...")
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("🎉 Perfect Match! (Strict=True) - 構造が完全に一致しました！")
    except Exception as e:
        print("❌ まだ不一致があります。エラーログを確認してください。")
        print(e)
        # 強制続行はしません（ゴミができるだけなので）
        return

    model.eval()

    # ダミー入力の作成
    # in_dim=35 に合わせた入力 (32ch 特徴量 + 3ch 座標)
    dummy_input = torch.randn(1, 35, 512, 512)
    
    # id_embedding (または style_code)
    # styleunet.py の forward の引数定義に依存しますが、
    # 通常は [1, 512] または [1, 256]
    # extra_style_dim=512 なので、ここも 512 かもしれません。
    # 一旦 512 で試します。エラーが出たら 256 にします。
    dummy_style = torch.randn(1, 512) 

    output_path = "neural_refiner.onnx"
    
    # ONNXエクスポート
    # ※ forwardの引数が合わない場合ここでエラーが出ます
    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_style),
            output_path,
            input_names=['fused_input', 'style_vector'], # TS側: concat(feature, uv) -> fused_input
            output_names=['neural_texture'], # 96chのテクスチャ
            opset_version=14
        )
        print(f"\n📦 生成完了: {output_path}")
        print("注意: TypeScript側で入力テンソルを [1, 35, 512, 512] に結合して渡してください。")
        
    except TypeError as e:
        print("\n❌ Forwardエラー: 引数の数や型が合っていません。")
        print(e)
        print("styleunet.py の forward メソッドの定義を確認してください。")

if __name__ == "__main__":
    export()