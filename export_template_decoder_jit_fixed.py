"""
Template Decoder ONNX Export with torch.jit.script Fix
Based on PyTorch Issue #28611 solution
Fixed to export with embedded data (no external .data file)
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from pathlib import Path
import io

# ======================================================================
# ✅ JIT Script Helper Function
# ======================================================================
@torch.jit.script
def get_base_features_jit(base_embedding_weight: torch.Tensor, N: int):
    """
    JIT-scripted function to get base features for N vertices
    This ensures ONNX export handles dynamic N correctly
    
    Args:
        base_embedding_weight: [num_vertices, 128] embedding weights
        N: Number of vertices to fetch (dynamic)
    
    Returns:
        [N, 128] base features
    """
    vertex_ids = torch.arange(N, dtype=torch.long)
    # Use embedding lookup via indexing
    return base_embedding_weight[vertex_ids]


# ======================================================================
# Model Components
# ======================================================================

class GlobalFeatureMapping(nn.Module):
    """Maps 768-dim global embedding to 256-dim"""
    def __init__(self):
        super().__init__()
        # ✅ 修正: 768→256→256→256 の3層構造
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
        
        # ✅ 修正: Feature layers: 512→256→256→256→256 (4層)
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
        
        # ✅ 修正: 各属性ヘッドの入力は 283 (256 + 27)
        # 256: feature_layers出力
        # 27: view_dirs
        self.color_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0 ← 283に修正
            nn.ReLU(),            # 1
            nn.Linear(128, 32)    # 2 - RGB SH coefficients
        )
        
        self.opacity_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0 ← 283に修正
            nn.ReLU(),            # 1
            nn.Linear(128, 1)     # 2 - Opacity
        )
        
        self.scale_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0 ← 283に修正
            nn.ReLU(),            # 1
            nn.Linear(128, 3)     # 2 - Scale
        )
        
        self.rotation_layers = nn.Sequential(
            nn.Linear(283, 128),  # 0 ← 283に修正
            nn.ReLU(),            # 1
            nn.Linear(128, 4)     # 2 - Rotation quaternion
        )
    
    def forward(self, fused_features, view_dirs):
        # fused_features: [B, N, 512]
        # view_dirs: [B, 27]
        
        # Shared feature extraction
        x = self.feature_layers(fused_features)  # [B, N, 256]
        
        # ✅ 修正: view_dirsを各頂点に拡張して連結
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
# ✅ Complete Template Decoder with JIT Script Fix
# ======================================================================

class CompleteTemplateDecoderWithJIT(nn.Module):
    """
    Complete Template Decoder with JIT Script fix for dynamic slicing
    
    This version uses @torch.jit.script to handle dynamic vertex indexing,
    solving the ONNX export issue with dynamic slicing.
    """
    
    def __init__(self, global_mapping, vertex_base_feature, gs_decoder):
        super().__init__()
        
        self.global_mapping = global_mapping
        
        # ✅ Store embedding weights as buffer (not nn.Embedding)
        # This allows JIT script to access it directly
        num_vertices, feature_dim = vertex_base_feature.shape
        self.register_buffer('base_embedding_weight', vertex_base_feature)
        
        self.gs_decoder = gs_decoder
        
        print(f'[TemplateDecoder] ✅ Created with JIT script fix')
        print(f'  Base features: {num_vertices} vertices x {feature_dim} dims')
    
    def forward(self, projection_features, global_embedding_768, view_dirs):
        """
        Forward pass with JIT-scripted base feature extraction
        
        Args:
            projection_features: [B, N, 128]
            global_embedding_768: [B, 768]
            view_dirs: [B, 27]
        
        Returns:
            rgb, opacity, scale, rotation, offset, id_embedding_256
        """
        B, N, _ = projection_features.shape
        
        # 1. Global feature mapping: 768 → 256
        global_feature_256 = self.global_mapping(global_embedding_768)  # [B, 256]
        global_expanded = global_feature_256.unsqueeze(1).expand(B, N, 256)  # [B, N, 256]
        
        # 2. ✅ Get base features using JIT script (handles dynamic N)
        base_for_batch = get_base_features_jit(self.base_embedding_weight, N)  # [N, 128]
        base_expanded = base_for_batch.unsqueeze(0).expand(B, N, 128)  # [B, N, 128]
        
        # 3. Concatenate features: projection + base + global
        fused_features = torch.cat([
            projection_features,  # [B, N, 128]
            base_expanded,        # [B, N, 128]
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
    print('🚀 Template Decoder Export with torch.jit.script Fix')
    print('=' * 70)
    print('✅ Solution: Using @torch.jit.script for dynamic slicing')
    print('✅ Based on: PyTorch Issue #28611')
    print('✅ Export: Embedded data (no external .data file)')
    print('=' * 70)
    
    # ──────────────────────────────────────────────────────────────────
    # Step 1: Load checkpoint
    # ──────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading checkpoint...')
    checkpoint_path = 'best_160000.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']  # ✅ Fixed: use checkpoint['model']
    print(f'✅ Loaded {len(state_dict)} keys')
    
    # ──────────────────────────────────────────────────────────────────
    # Step 2: Extract global_feature_mapping (3-layer: 0, 2, 4)
    # ──────────────────────────────────────────────────────────────────
    print('\n[2/5] Extracting global_feature_mapping...')
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
    print('\n[3/5] Extracting vertex_base_feature...')
    vertex_base_feature = state_dict['vertex_base_feature']
    print(f'   ✅ Base features: {list(vertex_base_feature.shape)}')
    
    # ──────────────────────────────────────────────────────────────────
    # Step 4: Extract vertex_gs_decoder
    # ──────────────────────────────────────────────────────────────────
    print('\n[4/5] Extracting vertex_gs_decoder...')
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
    # Step 5: Build complete model with JIT fix
    # ──────────────────────────────────────────────────────────────────
    print('\n[5/5] Building model with JIT script fix...')
    model = CompleteTemplateDecoderWithJIT(
        global_mapping,
        vertex_base_feature,
        gs_decoder
    )
    model.eval()
    print('   ✅ Complete model assembled (with JIT script)')
    
    # ──────────────────────────────────────────────────────────────────
    # Test before export
    # ──────────────────────────────────────────────────────────────────
    print('\n🧪 Testing model BEFORE ONNX export...')
    
    with torch.no_grad():
        for test_N in [100, 10595]:
            proj = torch.randn(1, test_N, 128)
            glob = torch.randn(1, 768)
            view = torch.zeros(1, 27)
            
            rgb, opacity, scale, rotation, offset, id_emb = model(proj, glob, view)
            
            opacity_np = opacity.squeeze().numpy()
            print(f'   Test N={test_N}:')
            print(f'     Opacity: min={opacity_np.min():.6f}, max={opacity_np.max():.6f}')
            print(f'     Unique opacity values: {len(np.unique(opacity_np))}')
    
    # ──────────────────────────────────────────────────────────────────
    # Export to ONNX with embedded data (no external .data file)
    # ──────────────────────────────────────────────────────────────────
    print('\n📦 Exporting to ONNX with embedded data (no external .data file)...')
    
    dummy_N = 100
    dummy_inputs = (
        torch.randn(1, dummy_N, 128),   # projection_features
        torch.randn(1, 768),             # global_embedding
        torch.zeros(1, 27)               # view_dirs
    )
    
    # ✅ 修正: BytesIOを使用して外部データファイルを回避
    f = io.BytesIO()
    
    torch.onnx.export(
        model,
        dummy_inputs,
        f,  # ✅ ファイルパスではなくBytesIOを使用
        input_names=['projection_features', 'global_embedding', 'view_dirs'],
        output_names=['rgb', 'opacity', 'scale', 'rotation', 'offset', 'id_embedding_256'],
        dynamic_axes={
            'projection_features': {1: 'N'},
            'rgb': {1: 'N'},
            'opacity': {1: 'N'},
            'scale': {1: 'N'},
            'rotation': {1: 'N'},
            'offset': {1: 'N'},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )
    
    # ✅ BytesIOから書き出し（外部データなし）
    with open('template_decoder.onnx', 'wb') as onnx_file:
        onnx_file.write(f.getvalue())
    
    onnx_path = Path('template_decoder.onnx')
    print(f'✅ Export complete: {onnx_path.name}')
    print(f'   Size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB')
    print(f'   ✅ All data embedded (no external .data file)')
    
    # ──────────────────────────────────────────────────────────────────
    # Verify with ONNX Runtime
    # ──────────────────────────────────────────────────────────────────
    print('\n🔍 Verifying with ONNX Runtime...')
    
    session = ort.InferenceSession('template_decoder.onnx')
    
    print('   Testing different N values:')
    for test_N in [50, 100, 1000, 10595]:
        proj_np = np.random.randn(1, test_N, 128).astype(np.float32)
        glob_np = np.random.randn(1, 768).astype(np.float32)
        view_np = np.zeros((1, 27), dtype=np.float32)
        
        feeds = {
            'projection_features': proj_np,
            'global_embedding': glob_np,
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
    print('✅ Export Complete with JIT Script Fix!')
    print('=' * 70)
    print('📋 Key Changes:')
    print('   ❌ OLD: vertex_ids = torch.arange(N)')
    print('   ❌ OLD: base = self.base_embedding(vertex_ids)')
    print('   ✅ NEW: @torch.jit.script def get_base_features_jit(...)')
    print('   ✅ NEW: base = get_base_features_jit(weights, N)')
    print('   ✅ NEW: All data embedded (no external .data file)')
    print('\n📝 Next steps:')
    print('   1. cp template_decoder.onnx public/assets/')
    print('   2. Hard reload browser (Ctrl+Shift+R)')
    print('   3. Check console: Opacity should have 10000+ unique values!')
    print('=' * 70)


if __name__ == '__main__':
    main()
