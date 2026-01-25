import torch
import os

WEIGHTS_PATH = "best_160000.pt"

if not os.path.exists(WEIGHTS_PATH):
    print("❌ ファイルがありません。")
    exit()

print(f"📂 Loading {WEIGHTS_PATH}...")
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

# 辞書の中身を取り出す
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("Type: state_dict found")
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
    print("Type: model dict found")
else:
    state_dict = checkpoint
    print("Type: raw dict")

print(f"\n🔑 Total keys: {len(state_dict)}")
print("First 50 keys sample:")
print("-" * 50)

# 最初の50個のキーを表示して、プレフィックスを確認する
count = 0
for k in state_dict.keys():
    print(k)
    count += 1
    if count >= 50:
        break

print("-" * 50)
print("検索ヒント:")
# 'mlp' や 'decoder' を含むキーを探す
mlp_keys = [k for k in state_dict.keys() if "mlp" in k]
decoder_keys = [k for k in state_dict.keys() if "decoder" in k]

print(f"Keys containing 'mlp': {len(mlp_keys)} (Example: {mlp_keys[0] if mlp_keys else 'None'})")
print(f"Keys containing 'decoder': {len(decoder_keys)} (Example: {decoder_keys[0] if decoder_keys else 'None'})")