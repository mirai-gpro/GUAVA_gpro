"""
Export Template Decoder Weights for WebGPU Implementation

Exports all FC layer weights to a single binary file that can be loaded
by the TypeScript WebGPU implementation.

Output files:
- base_features.bin (already exists from previous exports)
- template_decoder_weights.bin (new - all FC layer weights)
"""

import torch
import numpy as np
from pathlib import Path


def main():
    print('=' * 70)
    print('🚀 Export Template Decoder Weights for WebGPU')
    print('=' * 70)
    
    # Load checkpoint
    print('\n[1/3] Loading checkpoint...')
    checkpoint_path = 'best_160000.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    print(f'✅ Loaded {len(state_dict)} keys')
    
    # Collect weights in order
    print('\n[2/3] Extracting weights...')
    
    weights = []
    weight_info = []
    
    def add_weight(name: str, key: str, expected_shape: tuple):
        if key not in state_dict:
            raise ValueError(f'Key not found: {key}')
        tensor = state_dict[key]
        if tensor.shape != torch.Size(expected_shape):
            raise ValueError(f'Shape mismatch for {key}: {tensor.shape} vs {expected_shape}')
        arr = tensor.numpy().astype(np.float32).flatten()
        weights.append(arr)
        weight_info.append(f'  {name}: {key} {list(expected_shape)} = {arr.size} floats')
        return arr.size
    
    total_floats = 0
    
    # Global Feature Mapping (768→256→256→256)
    print('   Global Feature Mapping:')
    total_floats += add_weight('global_fc0_weight', 'global_feature_mapping.0.weight', (256, 768))
    total_floats += add_weight('global_fc0_bias', 'global_feature_mapping.0.bias', (256,))
    total_floats += add_weight('global_fc2_weight', 'global_feature_mapping.2.weight', (256, 256))
    total_floats += add_weight('global_fc2_bias', 'global_feature_mapping.2.bias', (256,))
    total_floats += add_weight('global_fc4_weight', 'global_feature_mapping.4.weight', (256, 256))
    total_floats += add_weight('global_fc4_bias', 'global_feature_mapping.4.bias', (256,))
    
    # Feature Layers (512→256→256→256→256)
    print('   Feature Layers:')
    total_floats += add_weight('feature_0_weight', 'vertex_gs_decoder.feature_layers.0.weight', (256, 512))
    total_floats += add_weight('feature_0_bias', 'vertex_gs_decoder.feature_layers.0.bias', (256,))
    total_floats += add_weight('feature_2_weight', 'vertex_gs_decoder.feature_layers.2.weight', (256, 256))
    total_floats += add_weight('feature_2_bias', 'vertex_gs_decoder.feature_layers.2.bias', (256,))
    total_floats += add_weight('feature_4_weight', 'vertex_gs_decoder.feature_layers.4.weight', (256, 256))
    total_floats += add_weight('feature_4_bias', 'vertex_gs_decoder.feature_layers.4.bias', (256,))
    total_floats += add_weight('feature_6_weight', 'vertex_gs_decoder.feature_layers.6.weight', (256, 256))
    total_floats += add_weight('feature_6_bias', 'vertex_gs_decoder.feature_layers.6.bias', (256,))
    
    # Color Head (283→128→32)
    print('   Color Head:')
    total_floats += add_weight('color_0_weight', 'vertex_gs_decoder.color_layers.0.weight', (128, 283))
    total_floats += add_weight('color_0_bias', 'vertex_gs_decoder.color_layers.0.bias', (128,))
    total_floats += add_weight('color_2_weight', 'vertex_gs_decoder.color_layers.2.weight', (32, 128))
    total_floats += add_weight('color_2_bias', 'vertex_gs_decoder.color_layers.2.bias', (32,))
    
    # Opacity Head (283→128→1)
    print('   Opacity Head:')
    total_floats += add_weight('opacity_0_weight', 'vertex_gs_decoder.opacity_layers.0.weight', (128, 283))
    total_floats += add_weight('opacity_0_bias', 'vertex_gs_decoder.opacity_layers.0.bias', (128,))
    total_floats += add_weight('opacity_2_weight', 'vertex_gs_decoder.opacity_layers.2.weight', (1, 128))
    total_floats += add_weight('opacity_2_bias', 'vertex_gs_decoder.opacity_layers.2.bias', (1,))
    
    # Scale Head (283→128→3)
    print('   Scale Head:')
    total_floats += add_weight('scale_0_weight', 'vertex_gs_decoder.scale_layers.0.weight', (128, 283))
    total_floats += add_weight('scale_0_bias', 'vertex_gs_decoder.scale_layers.0.bias', (128,))
    total_floats += add_weight('scale_2_weight', 'vertex_gs_decoder.scale_layers.2.weight', (3, 128))
    total_floats += add_weight('scale_2_bias', 'vertex_gs_decoder.scale_layers.2.bias', (3,))
    
    # Rotation Head (283→128→4)
    print('   Rotation Head:')
    total_floats += add_weight('rotation_0_weight', 'vertex_gs_decoder.rotation_layers.0.weight', (128, 283))
    total_floats += add_weight('rotation_0_bias', 'vertex_gs_decoder.rotation_layers.0.bias', (128,))
    total_floats += add_weight('rotation_2_weight', 'vertex_gs_decoder.rotation_layers.2.weight', (4, 128))
    total_floats += add_weight('rotation_2_bias', 'vertex_gs_decoder.rotation_layers.2.bias', (4,))
    
    print('\n   Weight extraction summary:')
    for info in weight_info:
        print(info)
    
    # Concatenate all weights
    all_weights = np.concatenate(weights)
    print(f'\n   Total: {total_floats} floats = {total_floats * 4 / 1024 / 1024:.2f} MB')
    
    # Save to binary file
    print('\n[3/3] Saving weights...')
    
    output_path = Path('template_decoder_weights.bin')
    all_weights.tofile(output_path)
    print(f'✅ Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)')
    
    # Also export base_features if not already done
    base_features = state_dict['vertex_base_feature'].numpy().astype(np.float32)
    base_path = Path('base_features.bin')
    base_features.tofile(base_path)
    print(f'✅ Saved: {base_path} ({base_path.stat().st_size / 1024 / 1024:.2f} MB)')
    
    # Verification
    print('\n🔍 Verification...')
    loaded = np.fromfile(output_path, dtype=np.float32)
    print(f'   Loaded {len(loaded)} floats')
    print(f'   Match: {len(loaded) == total_floats}')
    
    # Test: verify first weight matches
    first_weight = state_dict['global_feature_mapping.0.weight'].numpy().flatten()
    loaded_first = loaded[:first_weight.size]
    match = np.allclose(first_weight, loaded_first)
    print(f'   First weight verification: {match}')
    
    print('\n' + '=' * 70)
    print('✅ Export Complete!')
    print('=' * 70)
    print('\n📝 Next steps:')
    print('   1. cp template_decoder_weights.bin public/assets/')
    print('   2. cp base_features.bin public/assets/')
    print('   3. Replace template-decoder-onnx.ts with template-decoder-webgpu.ts')
    print('   4. Hard reload browser (Ctrl+Shift+R)')
    print('=' * 70)


if __name__ == '__main__':
    main()
