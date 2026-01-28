# convert_uv_point_decoder.py (正しい構造版)
import torch
import torch.nn as nn
import os

class UVPointDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature conv: 155 → 128 → 128 → 128
        self.feature_conv = nn.Sequential(
            nn.Conv2d(155, 128, 3, padding=1),  # 0
            nn.ReLU(),                           # 1
            nn.Conv2d(128, 128, 3, padding=1),  # 2
            nn.ReLU(),                           # 3
            nn.Conv2d(128, 128, 3, padding=1),  # 4
            nn.ReLU()                            # 5
        )
        
        # Local position head: 128 → 128 → 64 → 3
        self.local_pos_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 0
            nn.ReLU(),                           # 1
            nn.Conv2d(128, 64, 3, padding=1),   # 2
            nn.ReLU(),                           # 3
            nn.Conv2d(64, 3, 1, padding=0)      # 4
        )
        
        # Opacity head: 128 → 64 → 1
        self.opacity_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),   # 0
            nn.ReLU(),                           # 1
            nn.Conv2d(64, 1, 1, padding=0)      # 2
        )
        
        # Scale head: 128 → 64 → 3
        self.scale_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),   # 0
            nn.ReLU(),                           # 1
            nn.Conv2d(64, 3, 1, padding=0)      # 2
        )
        
        # Rotation head: 128 → 64 → 4
        self.rot_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),   # 0
            nn.ReLU(),                           # 1
            nn.Conv2d(64, 4, 1, padding=0)      # 2
        )
        
        # Color head: 128 → 128 → 32
        self.color_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 0
            nn.ReLU(),                           # 1
            nn.Conv2d(128, 32, 1, padding=0)    # 2
        )
    
    def forward(self, uv_features):
        # uv_features: [B, 155, H, W]
        x = self.feature_conv(uv_features)  # [B, 128, H, W]
        
        local_pos = self.local_pos_head(x)  # [B, 3, H, W]
        opacity = self.opacity_head(x)      # [B, 1, H, W]
        scale = self.scale_head(x)          # [B, 3, H, W]
        rotation = self.rot_head(x)         # [B, 4, H, W]
        color = self.color_head(x)          # [B, 32, H, W]
        
        return local_pos, opacity, scale, rotation, color

print("="*60)
print("Converting UV Point Decoder to ONNX")
print("="*60)

checkpoint = torch.load("best_160000.pt", map_location="cpu")
state_dict = checkpoint['model']

model = UVPointDecoder()

# Load weights
prefix = 'uv_point_decoder.'
new_state = {}
for k, v in state_dict.items():
    if k.startswith(prefix):
        short_key = k[len(prefix):]
        new_state[short_key] = v

result = model.load_state_dict(new_state, strict=True)
print(f"✅ Loaded {len(new_state)} weights")

model.eval()

# ONNX export
dummy_input = torch.randn(1, 155, 64, 64, dtype=torch.float32)

print("\nExporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "uv_point_decoder.onnx",
    export_params=True,
    opset_version=17,
    input_names=['uv_features'],
    output_names=['local_pos', 'opacity', 'scale', 'rotation', 'color'],
    dynamic_axes={
        'uv_features': {0: 'batch', 2: 'height', 3: 'width'},
        'local_pos': {0: 'batch', 2: 'height', 3: 'width'},
        'opacity': {0: 'batch', 2: 'height', 3: 'width'},
        'scale': {0: 'batch', 2: 'height', 3: 'width'},
        'rotation': {0: 'batch', 2: 'height', 3: 'width'},
        'color': {0: 'batch', 2: 'height', 3: 'width'}
    },
    dynamo=False
)

size_mb = os.path.getsize('uv_point_decoder.onnx')/1024/1024
print(f"✅ uv_point_decoder.onnx exported ({size_mb:.2f} MB)")

# Test
print("\nTesting model...")
with torch.no_grad():
    local_pos, opacity, scale, rotation, color = model(dummy_input)
    print(f"  Input:      {dummy_input.shape}")
    print(f"  local_pos:  {local_pos.shape}")
    print(f"  opacity:    {opacity.shape}")
    print(f"  scale:      {scale.shape}")
    print(f"  rotation:   {rotation.shape}")
    print(f"  color:      {color.shape}")
print("✅ Test passed!")