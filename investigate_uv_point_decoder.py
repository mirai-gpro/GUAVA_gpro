# investigate_uv_point_decoder.py
import torch

checkpoint = torch.load("best_160000.pt", map_location="cpu")
state_dict = checkpoint['model']

print("="*60)
print("UV Point Decoder structure:")
print("="*60)

uv_keys = [k for k in state_dict.keys() if 'uv_point_decoder' in k]
for key in sorted(uv_keys):
    shape = state_dict[key].shape
    print(f"{key}: {shape}")