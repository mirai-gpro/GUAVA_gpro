# convert_uv_style_mapping.py
# uv_style_mapping (768→512) を単独でONNXエクスポート

import torch
import torch.nn as nn
import os

print("="*60)
print("Converting uv_style_mapping to ONNX")
print("="*60)

# Load checkpoint
checkpoint = torch.load("best_160000.pt", map_location="cpu")
state_dict = checkpoint['model']

# Check keys
uv_style_keys = [k for k in state_dict.keys() if 'uv_style_mapping' in k]
print(f"Found {len(uv_style_keys)} keys:")
for k in uv_style_keys:
    print(f"  {k}: {state_dict[k].shape}")

# Define model
class UVStyleMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.layers(x)

print("\nCreating model...")
style_mapping = UVStyleMapping()

print("Loading weights...")
style_mapping_state = {
    'layers.0.weight': state_dict['uv_style_mapping.0.weight'],
    'layers.0.bias': state_dict['uv_style_mapping.0.bias'],
    'layers.2.weight': state_dict['uv_style_mapping.2.weight'],
    'layers.2.bias': state_dict['uv_style_mapping.2.bias'],
    'layers.4.weight': state_dict['uv_style_mapping.4.weight'],
    'layers.4.bias': state_dict['uv_style_mapping.4.bias'],
}
style_mapping.load_state_dict(style_mapping_state)
style_mapping.eval()
print("✅ Weights loaded")

# Test
print("\nTesting model...")
dummy_input = torch.randn(1, 768, dtype=torch.float32)
with torch.no_grad():
    output = style_mapping(dummy_input)
print(f"  Input:  {dummy_input.shape}")
print(f"  Output: {output.shape}")

# Export
print("\nExporting to ONNX...")
torch.onnx.export(
    style_mapping,
    dummy_input,
    "uv_style_mapping.onnx",
    export_params=True,
    opset_version=14,
    input_names=['global_feature'],
    output_names=['extra_style'],
    dynamic_axes={
        'global_feature': {0: 'batch'},
        'extra_style': {0: 'batch'}
    },
    dynamo=False
)

size_mb = os.path.getsize('uv_style_mapping.onnx') / 1024 / 1024
print(f"✅ uv_style_mapping.onnx exported ({size_mb:.2f} MB)")
