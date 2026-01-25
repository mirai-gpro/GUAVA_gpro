#!/usr/bin/env python3
"""
convert_styleunet_to_onnx_v2.py
Convert the REAL GUAVA Neural Refiner (StyleUNet) to ONNX

The refiner weights are stored at: ck['render_model']['nerual_refiner.*']

Usage:
  python convert_styleunet_to_onnx_v2.py best_160000.pt styleunet_refiner.onnx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os


# ============================================================
# Model definitions from styleunet.py
# ============================================================

class NormStyleCode(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ConstantInput(nn.Module):
    def __init__(self, num_channel, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        skip = self.skip(x)
        out = out + skip
        return out


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, 
                 demodulate=True, sample_mode=None, eps=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size) /
            math.sqrt(in_channels * kernel_size**2)
        )
        self.padding = kernel_size // 2

    def forward(self, x, style):
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        
        if self.sample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif self.sample_mode == 'downsample':
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:4])
        return out


class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, 
                 demodulate=True, sample_mode=None):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat, 
            demodulate=demodulate, sample_mode=sample_mode
        )
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style, noise=None):
        out = self.modulated_conv(x, style) * 2 ** 0.5
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        out = out + self.bias
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, num_style_feat, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size=1, 
            num_style_feat=num_style_feat, demodulate=False, sample_mode=None
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return out


class StyleGAN2GeneratorCSFT_small(nn.Module):
    def __init__(self, out_size, out_dim=3, num_style_feat=512, num_mlp=8, channel_scale=1):
        super().__init__()
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        for key in channels.keys():
            channels[key] = int(channels[key] / channel_scale)
        self.channels = channels
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 1 + 1
        self.num_latent = self.log_size * 1
        
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.extend([
                nn.Linear(num_style_feat, num_style_feat, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'], channels['4'], kernel_size=3,
            num_style_feat=num_style_feat, demodulate=True, sample_mode=None
        )
        self.to_rgb1 = ToRGB(channels['4'], out_dim, num_style_feat, upsample=False)
        
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        self.normal_convs = nn.ModuleList()
        in_channels = channels['4']
        
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels, out_channels, kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True, sample_mode='upsample'
                )
            )
            self.normal_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
            self.to_rgbs.append(
                ToRGB(out_channels, out_dim, num_style_feat, upsample=True)
            )
            in_channels = out_channels

    def forward(self, styles, conditions, randomize_noise=False):
        styles = self.style_mlp(styles)
        
        if randomize_noise:
            noise = [None] * self.num_layers
        else:
            noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        
        inject_index = self.num_latent
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles
        
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        
        for conv1, conv2, noise1, to_rgb in zip(
                self.style_convs, self.normal_convs,
                noise[1:], self.to_rgbs
            ):
            out = conv1(out, latent[:, i], noise=noise1)
            if i < len(conditions):
                out = out * conditions[(i-1)*2] + conditions[(i-1)*2+1]
            out = conv2(out)
            skip = to_rgb(out, latent[:, i+1], skip)
            i += 1
        image = skip
        return image


class StyleUNet(nn.Module):
    def __init__(
        self, in_size, out_size, in_dim, out_dim, 
        num_style_feat=512, num_mlp=8, activation=True,
        channel_scale=1, small=False, extra_style_dim=-1,
    ): 
        super().__init__()
        
        self.activation = activation
        self.num_style_feat = num_style_feat
        self.in_size, self.out_size = in_size, out_size
        channels = {
            '4': 256, '8': 256, '16': 256, '32': 256,
            '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8
        }
        for key in channels.keys():
            channels[key] = int(channels[key] / channel_scale)
        
        self.log_size = int(math.log(out_size, 2))
        
        if self.in_size <= self.out_size:
            self.conv_body_first = nn.Conv2d(in_dim, channels[f'{out_size}'], 1)
        else:
            self.conv_body_first = nn.ModuleList([
                nn.Conv2d(in_dim, channels[f'{in_size}'], 1),
                ResBlock(channels[f'{in_size}'], channels[f'{out_size}'], mode='down'),
            ])
        
        in_channels = channels[f'{out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)
        
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        self.final_linear = nn.Linear(channels['4'] * 4 * 4, num_style_feat)
        self.extra_style_dim = extra_style_dim
        if extra_style_dim > 0:
            self.style_fuse = nn.Sequential(
                nn.Linear(extra_style_dim + num_style_feat, num_style_feat),
                nn.LeakyReLU(0.2, True),
                nn.Linear(num_style_feat, num_style_feat),
            )
        
        if small:
            self.stylegan_decoder = StyleGAN2GeneratorCSFT_small(
                out_dim=out_dim, out_size=out_size, 
                num_style_feat=num_style_feat, num_mlp=num_mlp,
                channel_scale=channel_scale,
            )
        else:
            raise NotImplementedError("Only small variant supported")
        
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            ch = channels[f'{2**i}']
            self.condition_scale.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 3, 1, 1)
            ))
            self.condition_shift.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 3, 1, 1)
            ))

    def forward(self, x, randomize_noise=False, extra_style=None):
        conditions, unet_skips = [], []
        
        if x.shape[-1] < self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        
        if self.in_size <= self.out_size:
            feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        else:
            feat = F.leaky_relu_(self.conv_body_first[0](x), negative_slope=0.2)
            feat = self.conv_body_first[1](feat)
        
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)
        
        style_code = self.final_linear(feat.reshape(feat.size(0), -1))
        if self.extra_style_dim > 0 and extra_style is not None:
            style_code = self.style_fuse(torch.cat([style_code, extra_style], dim=1))
        
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())

        image = self.stylegan_decoder(style_code, conditions, randomize_noise=randomize_noise)
        
        if self.activation:
            image = torch.sigmoid(image)
           
        return image


# ============================================================
# Main conversion
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_styleunet_to_onnx_v2.py <checkpoint.pt> <output.onnx>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("=" * 70)
    print("🚀 StyleUNet (Neural Refiner) ONNX Converter v2")
    print("=" * 70)
    
    # Build model with config from ubody_512.yaml
    print("\n📦 Building StyleUNet model...")
    model = StyleUNet(
        in_size=512, 
        out_size=512,
        in_dim=32,
        out_dim=3,
        num_style_feat=512,
        num_mlp=8,
        activation=True,
        channel_scale=1,
        small=True,
        extra_style_dim=-1,
    )
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract render_model -> nerual_refiner weights
    if 'render_model' not in checkpoint:
        print("❌ 'render_model' key not found in checkpoint!")
        sys.exit(1)
    
    render_model_state = checkpoint['render_model']
    
    # Filter nerual_refiner weights and remove prefix
    refiner_state = {}
    for k, v in render_model_state.items():
        if k.startswith('nerual_refiner.'):
            new_k = k.replace('nerual_refiner.', '')
            refiner_state[new_k] = v
    
    print(f"  Found {len(refiner_state)} nerual_refiner tensors")
    
    if len(refiner_state) == 0:
        print("❌ No 'nerual_refiner' weights found!")
        sys.exit(1)
    
    # Load weights
    missing, unexpected = model.load_state_dict(refiner_state, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    if missing:
        print("  ⚠️  Some missing keys (first 10):")
        for k in missing[:10]:
            print(f"      {k}")
    
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Test inference
    print("\n🔬 Testing PyTorch inference...")
    dummy_input = torch.randn(1, 32, 512, 512)
    with torch.no_grad():
        # Use randomize_noise=True to avoid noise buffer size mismatch
        output = model(dummy_input, randomize_noise=True)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # ONNX Export
    print("\n📦 Exporting to ONNX...")
    print("   ⚠️  Note: ModulatedConv2d uses dynamic operations")
    
    # Create a wrapper that always uses randomize_noise=True
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x, randomize_noise=True)
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["rendered_features"],
            output_names=["rgb_output"],
            dynamic_axes={
                "rendered_features": {0: "batch"},
                "rgb_output": {0: "batch"},
            },
            do_constant_folding=True,
        )
        
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print("\n" + "=" * 70)
        print("✅ EXPORT SUCCESSFUL!")
        print("=" * 70)
        print(f"📄 File: {output_path}")
        print(f"💾 Size: {size_mb:.2f} MB")
        print(f"📐 Input: [batch, 32, 512, 512]")
        print(f"📐 Output: [batch, 3, 512, 512]")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        print("\n" + "=" * 70)
        print("Alternative: Extract weights for WebGL implementation")
        print("=" * 70)
        
        # Save weights as JSON for WebGL
        import json
        import base64
        
        weights_dict = {}
        for k, v in refiner_state.items():
            arr = v.detach().cpu().float().numpy()
            weights_dict[k] = base64.b64encode(arr.tobytes()).decode('utf-8')
        
        json_path = output_path.replace('.onnx', '_weights.json')
        with open(json_path, 'w') as f:
            json.dump(weights_dict, f)
        
        json_size_mb = os.path.getsize(json_path) / 1024 / 1024
        print(f"📄 Weights JSON: {json_path}")
        print(f"💾 Size: {json_size_mb:.2f} MB")


if __name__ == '__main__':
    main()