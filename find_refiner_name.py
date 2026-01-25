import torch
import os

WEIGHTS_PATH = "best_160000.pt"

if not os.path.exists(WEIGHTS_PATH):
    print("❌ ファイルがありません。")
    exit()

print(f"🔍 Refinerの名前を捜索中...")
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# Refinerにしか存在しない特徴的なレイヤー名で検索
keywords = ["conv_body_first", "to_rgb", "style_mlp", "id_mlp"]
found_prefixes = set()

print("\n--- 検索結果 ---")
for key in state_dict.keys():
    for kw in keywords:
        if kw in key:
            # "rgb_decoder.conv_body_first.weight" -> "rgb_decoder" を抽出
            # "net.styleunet.conv_body_first.weight" -> "net.styleunet" を抽出
            parts = key.split(kw)
            prefix = parts[0].rstrip(".")
            found_prefixes.add(prefix)
            print(f"ヒット: {key}  --->  プレフィックス: '{prefix}'")
            break

print("\n" + "="*30)
if found_prefixes:
    print(f"✅ Refinerの正体判明: {list(found_prefixes)}")
else:
    print("⚠️ Refinerのキーワードが見つかりませんでした。")
    print("もしかして: このモデルにはRefinerが含まれていない？ (いや、サイズ的にあるはず)")
    # 念のため 'rgb' がつくキーを全部出す
    print("\n'rgb' を含むキー一覧:")
    for k in state_dict.keys():
        if "rgb" in k:
            print(k)

print("="*30)