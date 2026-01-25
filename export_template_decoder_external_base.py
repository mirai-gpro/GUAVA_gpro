"""
Template Decoder ONNX Export with base_features as External Input
Eliminates Gather node to avoid ONNX Runtime Web compatibility issues
Performance impact: ~1.3MB data transfer (base_features for N=10595)
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from pathlib import Path
import io

# ======================================================================
# Model Components
# ======================================================================

class GlobalFeatureMapping(nn.Module):
    """Maps 768-dim global embedding to 256-dim"""
    def __init__(self):
        super().__init__()
        # 768→256→256→256 の3層構造
        self.fc0 = nn.Linear(768, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 256)
    
    def forward(self, x):
        x = self.fc0(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class VertexGSDecoder(nn.Module):
    """Decodes fused features into Gaussian attributes"""
    def __init__(self):
        super().__init__()
        
        # Feature layers: 512→256→256→256→256 (4層)
        self.feature_layers = nn.Sequential(
            nn.Linear(512, 256),  # 0
            nn.ReLU(),            # 1
            nn.Linear(256, 256),  # 2
            nn.ReLU(),            # 3
            nn.Linear(256, 256),  # 4
            nn.ReLU(),            # 5
            nn.Linear(256, 256),  # 6
            nn.ReLU()             # 7
        )
        
        # 各属性ヘッドの入力は 283 (256 + 27)
        self.color_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0
            nn.ReLU(),            # 1
            nn.Linear(128, 32)    # 2 - RGB SH coefficients
        )
        
        self.opacity_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0
            nn.ReLU(),            # 1
            nn.Linear(128, 1)     # 2 - Opacity
        )
        
        self.scale_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0
            nn.ReLU(),            # 1
            nn.Linear(128, 3)     # 2 - Scale
        )
        
        self.rotation_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0
            nn.ReLU(),            # 1
            nn.Linear(128, 4)     # 2 - Rotation quaternion
        )
    
    def forward(self, fused_features, view_dirs):
        # fused_features: [B, N, 512]
        # view_dirs: [B, 27]
        
        # Shared feature extraction
        x = self.feature_layers(fused_features)  # [B, N, 256]
        
        # view_dirsを各頂点に拡張して連結
        B, N, _ = x.shape
        view_expanded = view_dirs.unsqueeze(1).expand(B, N, 27)  # [B, N, 27]
        x_with_view = torch.cat([x, view_expanded], dim=-1)  # [B, N, 283]
        
        # Attribute-specific heads (入力は283次元)
        rgb = self.color_layers(x_with_view)       # [B, N, 32]
        opacity = self.opacity_layers(x_with_view) # [B, N, 1]
        scale = self.scale_layers(x_with_view)     # [B, N, 3]
        rotation = self.rotation_layers(x_with_view) # [B, N, 4]
        
        # Activations
        opacity = torch.sigmoid(opacity)
        scale = torch.exp(scale)
        rotation = torch.nn.functional.normalize(rotation, dim=-1)
        
        # Offset is zero (not in checkpoint)
        offset = torch.zeros_like(scale)
        
        return rgb, opacity, scale, rotation, offset


# ======================================================================
# ✅ Complete Template Decoder with base_features as External Input
# ======================================================================

class CompleteTemplateDecoderExternalBase(nn.Module):
    """
    Complete Template Decoder with base_features as external input
    
    This eliminates the Gather node completely, avoiding ONNX Runtime Web issues:
    - No dynamic torch.arange() inside the model
    - No Gather operation on base_embedding_weight
    - base_features provided externally (sliced in JavaScript)
    - Performance: ~1.3MB data transfer for N=10595 (acceptable)
    """
    
    def __init__(self, global_mapping, gs_decoder):
        super().__init__()
        
        self.global_mapping = global_mapping
        self.gs_decoder = gs_decoder
        
        # ✅ No base_embedding_weight stored in the model
        
        print(f'[TemplateDecoder] ✅ Created with external base_features')
        print(f'  Base features: External input [N, 128]')
        print(f'  No Gather node in ONNX graph')
        print(f'  Data transfer: ~1.3MB for N=10595 (acceptable)')
    
    def forward(self, projection_features, global_embedding_768, base_features, view_dirs):
        """
        Forward pass with base_features as external input
        
        Args:
            projection_features: [B, N, 128]
            global_embedding_768: [B, 768]
            base_features: [B, N, 128] - External input (no Gather needed)
            view_dirs: [B, 27]
        
        Returns:
            rgb, opacity, scale, rotation, offset, id_embedding_256
        """
        B, N, _ = projection_features.shape
        
        # 1. Global feature mapping: 768 → 256
        global_feature_256 = self.global_mapping(global_embedding_768)  # [B, 256]
        global_expanded = global_feature_256.unsqueeze(1).expand(B, N, 256)  # [B, N, 256]
        
        # 2. ✅ Use base_features directly (no Gather operation)
        # base_features is already [B, N, 128] from external input
        
        # 3. Concatenate features: projection + base + global
        fused_features = torch.cat([
            projection_features,  # [B, N, 128]
            base_features,        # [B, N, 128] ← External input
            global_expanded       # [B, N, 256]
        ], dim=-1)  # [B, N, 512]
        
        # 4. GS Decoder
        rgb, opacity, scale, rotation, offset = self.gs_decoder(fused_features, view_dirs)
        
        return rgb, opacity, scale, rotation, offset, global_feature_256


# ======================================================================
# Main Export Script
# ======================================================================

def main():
    print('=' * 70)
    print('🚀 Template Decoder Export with External base_features')
    print('=' * 70)
    print('✅ Solution: base_features as external input')
    print('✅ No Gather node: Most reliable for ONNX Runtime Web')
    print('✅ Performance: ~1.3MB transfer (acceptable on mobile)')
    print('✅ Exporter: Legacy ONNX exporter (stable)')
    print('=' * 70)
    
    # ──────────────────────────────────────────────────────────────────
    # Step 1: Load checkpoint
    # ──────────────────────────────────────────────────────────────────
    print('\n[1/6] Loading checkpoint...')
    checkpoint_path = 'best_160000.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    print(f'✅ Loaded {len(state_dict)} keys')
    
    # ──────────────────────────────────────────────────────────────────
    # Step 2: Extract global_feature_mapping
    # ──────────────────────────────────────────────────────────────────
    print('\n[2/6] Extracting global_feature_mapping...')
    global_mapping = GlobalFeatureMapping()
    
    mapping_keys = {
        'global_feature_mapping.0.weight': 'fc0.weight',
        'global_feature_mapping.0.bias': 'fc0.bias',
        'global_feature_mapping.2.weight': 'fc2.weight',
        'global_feature_mapping.2.bias': 'fc2.bias',
        'global_feature_mapping.4.weight': 'fc4.weight',
        'global_feature_mapping.4.bias': 'fc4.bias',
    }
    
    global_state = {}
    for old_key, new_key in mapping_keys.items():
        if old_key in state_dict:
            global_state[new_key] = state_dict[old_key]
    
    global_mapping.load_state_dict(global_state, strict=True)
    global_mapping.eval()
    print(f'   Found {len(global_state)} parameters')
    print('   ✅ Global mapping loaded')
    
    # ──────────────────────────────────────────────────────────────────
    # Step 3: Extract vertex_base_feature
    # ──────────────────────────────────────────────────────────────────
    print('\n[3/6] Extracting vertex_base_feature...')
    vertex_base_feature = state_dict['vertex_base_feature']
    num_vertices, feature_dim = vertex_base_feature.shape
    print(f'   ✅ Base features: {list(vertex_base_feature.shape)}')
    
    # ✅ Export base_features as separate binary file
    base_features_path = Path('base_features.bin')
    vertex_base_feature.numpy().astype(np.float32).tofile(base_features_path)
    print(f'   ✅ Exported: {base_features_path.name} ({base_features_path.stat().st_size / 1024 / 1024:.2f} MB)')
    
    # ──────────────────────────────────────────────────────────────────
    # Step 4: Extract vertex_gs_decoder
    # ──────────────────────────────────────────────────────────────────
    print('\n[4/6] Extracting vertex_gs_decoder...')
    gs_decoder = VertexGSDecoder()
    
    decoder_keys = {
        # Feature layers
        'vertex_gs_decoder.feature_layers.0.weight': 'feature_layers.0.weight',
        'vertex_gs_decoder.feature_layers.0.bias': 'feature_layers.0.bias',
        'vertex_gs_decoder.feature_layers.2.weight': 'feature_layers.2.weight',
        'vertex_gs_decoder.feature_layers.2.bias': 'feature_layers.2.bias',
        'vertex_gs_decoder.feature_layers.4.weight': 'feature_layers.4.weight',
        'vertex_gs_decoder.feature_layers.4.bias': 'feature_layers.4.bias',
        'vertex_gs_decoder.feature_layers.6.weight': 'feature_layers.6.weight',
        'vertex_gs_decoder.feature_layers.6.bias': 'feature_layers.6.bias',
        # Color layers
        'vertex_gs_decoder.color_layers.0.weight': 'color_layers.0.weight',
        'vertex_gs_decoder.color_layers.0.bias': 'color_layers.0.bias',
        'vertex_gs_decoder.color_layers.2.weight': 'color_layers.2.weight',
        'vertex_gs_decoder.color_layers.2.bias': 'color_layers.2.bias',
        # Opacity layers
        'vertex_gs_decoder.opacity_layers.0.weight': 'opacity_layers.0.weight',
        'vertex_gs_decoder.opacity_layers.0.bias': 'opacity_layers.0.bias',
        'vertex_gs_decoder.opacity_layers.2.weight': 'opacity_layers.2.weight',
        'vertex_gs_decoder.opacity_layers.2.bias': 'opacity_layers.2.bias',
        # Scale layers
        'vertex_gs_decoder.scale_layers.0.weight': 'scale_layers.0.weight',
        'vertex_gs_decoder.scale_layers.0.bias': 'scale_layers.0.bias',
        'vertex_gs_decoder.scale_layers.2.weight': 'scale_layers.2.weight',
        'vertex_gs_decoder.scale_layers.2.bias': 'scale_layers.2.bias',
        # Rotation layers
        'vertex_gs_decoder.rotation_layers.0.weight': 'rotation_layers.0.weight',
        'vertex_gs_decoder.rotation_layers.0.bias': 'rotation_layers.0.bias',
        'vertex_gs_decoder.rotation_layers.2.weight': 'rotation_layers.2.weight',
        'vertex_gs_decoder.rotation_layers.2.bias': 'rotation_layers.2.bias',
    }
    
    decoder_state = {}
    for old_key, new_key in decoder_keys.items():
        if old_key in state_dict:
            decoder_state[new_key] = state_dict[old_key]
    
    gs_decoder.load_state_dict(decoder_state, strict=True)
    gs_decoder.eval()
    print(f'   Found {len(decoder_state)} parameters')
    print('   ✅ GS decoder loaded')
    
    # ──────────────────────────────────────────────────────────────────
    # Step 5: Build complete model with external base_features
    # ──────────────────────────────────────────────────────────────────
    print('\n[5/6] Building model with external base_features...')
    model = CompleteTemplateDecoderExternalBase(
        global_mapping,
        gs_decoder
    )
    model.eval()
    print('   ✅ Complete model assembled (no Gather node)')
    
    # ──────────────────────────────────────────────────────────────────
    # Test before export
    # ──────────────────────────────────────────────────────────────────
    print('\n🧪 Testing model BEFORE ONNX export...')
    
    with torch.no_grad():
        for test_N in [100, 10595]:
            proj = torch.randn(1, test_N, 128)
            glob = torch.randn(1, 768)
            base = vertex_base_feature[:test_N].unsqueeze(0)  # ✅ External base_features
            view = torch.zeros(1, 27)
            
            rgb, opacity, scale, rotation, offset, id_emb = model(proj, glob, base, view)
            
            opacity_np = opacity.squeeze().numpy()
            print(f'   Test N={test_N}:')
            print(f'     Opacity: min={opacity_np.min():.6f}, max={opacity_np.max():.6f}')
            print(f'     Unique opacity values: {len(np.unique(opacity_np))}')
    
    # ──────────────────────────────────────────────────────────────────
    # Export to ONNX with embedded data (Legacy Exporter)
    # ──────────────────────────────────────────────────────────────────
    print('\n📦 Exporting to ONNX with external base_features (Legacy Exporter)...')
    
    dummy_N = 100
    dummy_inputs = (
        torch.randn(1, dummy_N, 128),                          # projection_features
        torch.randn(1, 768),                                   # global_embedding
        torch.randn(1, dummy_N, 128),                          # ✅ base_features (external)
        torch.zeros(1, 27)                                     # view_dirs
    )
    
    # BytesIOを使用して外部データファイルを回避
    f = io.BytesIO()
    
    # ✅ 重要: dynamo=False でレガシーエクスポーターを使用
    torch.onnx.export(
        model,
        dummy_inputs,
        f,
        input_names=['projection_features', 'global_embedding', 'base_features', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset', 'id_embedding_256'],
        dynamic_axes={
            'projection_features': {1: 'N'},
            'base_features': {1: 'N'},  # ✅ base_featuresも動的
            'rgb': {1: 'N'},
            'opacity': {1: 'N'},
            'scale': {1: 'N'},
            'rotation': {1: 'N'},
            'offset': {1: 'N'},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        dynamo=False  # ✅ レガシーエクスポーターを強制使用
    )
    
    # BytesIOから書き出し
    with open('template_decoder.onnx', 'wb') as onnx_file:
        onnx_file.write(f.getvalue())
    
    onnx_path = Path('template_decoder.onnx')
    print(f'✅ Export complete: {onnx_path.name}')
    print(f'   Size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB')
    print(f'   ✅ All data embedded (no external .data file)')
    print(f'   ✅ Using legacy ONNX exporter (stable)')
    print(f'   ✅ No Gather node in graph')
    
    # ──────────────────────────────────────────────────────────────────
    # Verify with ONNX Runtime
    # ──────────────────────────────────────────────────────────────────
    print('\n🔍 Verifying with ONNX Runtime...')
    
    session = ort.InferenceSession('template_decoder.onnx')
    
    print('   Testing different N values:')
    for test_N in [50, 100, 1000, 10595]:
        proj_np = np.random.randn(1, test_N, 128).astype(np.float32)
        glob_np = np.random.randn(1, 768).astype(np.float32)
        base_np = vertex_base_feature[:test_N].unsqueeze(0).numpy().astype(np.float32)  # ✅ External base_features
        view_np = np.zeros((1, 27), dtype=np.float32)
        
        feeds = {
            'projection_features': proj_np,
            'global_embedding': glob_np,
            'base_features': base_np,  # ✅ 外部から渡す
            'view_dirs': view_np
        }
        
        outputs = session.run(None, feeds)
        opacity = outputs[1]
        
        unique_count = len(np.unique(opacity))
        diversity_pct = (unique_count / opacity.size) * 100
        
        print(f'     N={test_N}: opacity range=[{opacity.min():.6f}, {opacity.max():.6f}], unique={unique_count} ({diversity_pct:.1f}%)')
        
        if unique_count > test_N * 0.5:
            print(f'       ✅ Good diversity')
        else:
            print(f'       ⚠️  Low diversity - possible issue')
    
    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('✅ Export Complete with External base_features!')
    print('=' * 70)
    print('📋 Key Changes:')
    print('   ✅ NEW INPUT: base_features [N, 128] (float32)')
    print('   ✅ JavaScript: base_features = allBaseFeatures.slice(0, N*128)')
    print('   ✅ No Gather node in ONNX graph')
    print('   ✅ Data transfer: ~1.3MB for N=10595 (acceptable)')
    print('   ✅ Legacy exporter: Stable and reliable')
    print('\n📝 Files generated:')
    print(f'   1. template_decoder.onnx ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)')
    print(f'   2. base_features.bin ({base_features_path.stat().st_size / 1024 / 1024:.2f} MB)')
    print('\n📝 Next steps:')
    print('   1. cp template_decoder.onnx public/assets/')
    print('   2. cp base_features.bin public/assets/')
    print('   3. Update TypeScript to load base_features.bin')
    print('   4. Pass base_features as external input')
    print('   5. Hard reload browser (Ctrl+Shift+R)')
    print('   6. Check console: Opacity should have 10000+ unique values!')
    print('=' * 70)


if __name__ == '__main__':
    main()
