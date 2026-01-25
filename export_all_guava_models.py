import torch
import torch.nn as nn
import onnx
import sys
import os

# パス設定 (環境に合わせて書き換えてください)
sys.path.append(".") 

# 必要なモジュールをインポート
# ※エラーが出る場合はフォルダ構成を確認して修正してください
try:
    from models.modules.net_module.styleunet import StyleUNet
    from models.modules.net_module.feature_decoder import Vertex_GS_Decoder
    # DINO Encoderは外部ライブラリ依存が激しいので、今回はRefinerとDecoderを優先します
    print("✅ モジュール定義のロードに成功")
except ImportError as e:
    print(f"❌ モジュールが見つかりません: {e}")
    print("スクリプトをGUAVAフォルダのルートに置いて実行してください。")
    exit()

# ------------------------------------------------------------------
# 1. 重みファイルの確認
# ------------------------------------------------------------------
WEIGHTS_PATH = "best_160000.pt"

if not os.path.exists(WEIGHTS_PATH):
    print(f"❌ '{WEIGHTS_PATH}' が見つかりません。")
    print("ダウンロードした .pt ファイルをこのスクリプトと同じ場所に置いてください。")
    exit()

print(f"📂 重みをロード中: {WEIGHTS_PATH} (1.2GBあるので時間かかります...)")
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

# チェックポイントの構造を確認
state_dict = checkpoint
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']

print("✅ 重みのロード完了。キーの数:", len(state_dict))

# ------------------------------------------------------------------
# 2. Neural Refiner (StyleUNet + MLP) のエクスポート
# ------------------------------------------------------------------
print("\n🚀 [1/2] Neural Refiner (CompleteRefiner) のエクスポート...")

class CompleteRefiner(nn.Module):
    def __init__(self, id_dim=256, style_dim=512):
        super().__init__()
        # ubody_gaussian.py の構造を再現
        self.id_mlp = nn.Sequential(
            nn.Linear(id_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )
        self.rgb_decoder = StyleUNet(
            in_size=512, out_size=512, in_dim=32, out_dim=3, num_style_feat=512
        )

    def forward(self, feature_map, id_embedding):
        style = self.id_mlp(id_embedding)
        return self.rgb_decoder(feature_map, style)

refiner = CompleteRefiner()

# 重みの抽出とロード
refiner_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "")
    # ubody_gaussian.py では 'rgb_decoder' と 'id_mlp' という名前で保存されているはず
    if name.startswith("rgb_decoder") or name.startswith("id_mlp"):
        refiner_dict[name] = v

try:
    refiner.load_state_dict(refiner_dict, strict=True)
    print("✅ Refiner Weights Verified! (Strict Mode)")
except Exception as e:
    print("⚠️ Refiner Strict Load Failed. Trying loose load...", e)
    refiner.load_state_dict(refiner_dict, strict=False)

refiner.eval()
dummy_fm = torch.randn(1, 32, 512, 512)
dummy_id = torch.randn(1, 256)

torch.onnx.export(
    refiner, (dummy_fm, dummy_id), "neural_refiner_complete.onnx",
    input_names=["feature_map", "id_embedding"], output_names=["rgb_image"],
    opset_version=14
)
print("📦 neural_refiner_complete.onnx 生成完了！")


# ------------------------------------------------------------------
# 3. Template Decoder (Vertex_GS_Decoder) のエクスポート
# ------------------------------------------------------------------
print("\n🚀 [2/2] Template Decoder のエクスポート...")

# 初期化パラメータは ubody_gaussian.py の __init__ を参照
# 通常: in_dim=1024, dir_dim=27 (harmonic=4), color_out_dim=32
# feature_decoder.py の Vertex_GS_Decoder を使用
decoder = Vertex_GS_Decoder(in_dim=1024, dir_dim=27, color_out_dim=32)

# 重みの抽出
decoder_dict = {}
prefix = "prj_feature_decoder." # または "uv_feature_decoder"
for k, v in state_dict.items():
    name = k.replace("module.", "")
    if name.startswith(prefix):
        decoder_dict[name.replace(prefix, "")] = v

if len(decoder_dict) == 0:
    print("⚠️ 'prj_feature_decoder' が見つかりません。'uv_feature_decoder' で試します。")
    prefix = "uv_feature_decoder."
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        if name.startswith(prefix):
            decoder_dict[name.replace(prefix, "")] = v

try:
    decoder.load_state_dict(decoder_dict, strict=True)
    print("✅ Decoder Weights Verified! (Strict Mode)")
except Exception as e:
    print("⚠️ Decoder Strict Load Failed...", e)
    decoder.load_state_dict(decoder_dict, strict=False)

decoder.eval()

# 入力: [Projection(128) + Base(128) + Global(256)] = 512次元？
# ※注意: ubody_gaussian.py では Feature Concat 後に Vertex_GS_Decoder に入れますが
# Vertex_GS_Decoder の in_dim が 1024 なので、入力次元を確認する必要があります。
# ここでは一旦、実装されている通り 1024次元 の入力でエクスポートします。
# TypeScript側で cat([proj, base, global, ...]) で 1024 になるように調整が必要です。

dummy_feat = torch.randn(1, 10595, 1024) 
dummy_dir = torch.randn(1, 10595, 27) # view direction

torch.onnx.export(
    decoder, (dummy_feat, dummy_dir), "template_decoder.onnx",
    input_names=["fused_features", "view_dirs"], 
    output_names=["rgb", "opacity", "scale", "rotation", "offset"],
    opset_version=14
)
print("📦 template_decoder.onnx 生成完了！")

print("\n🎉 全工程終了。生成された .onnx ファイルを assets フォルダへ！")