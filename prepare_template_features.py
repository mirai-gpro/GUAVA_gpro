#!/usr/bin/env python3
"""
prepare_template_features.py

Template Decoder用の特徴を事前に連結してバイナリファイルとして出力するスクリプト

公式実装 (ubody_gaussian.py lines 126-133):
    vertex_base_feature = self.vertex_base_feature[None].expand(batch_size, -1, -1)
    vertex_sample_feature = self.sample_prj_feature(...)
    vertex_global_feature = self.global_feature_mapping(global_feature)
    vertex_global_feature = vertex_global_feature[:,None,:].expand(-1, N, -1)
    
    # 連結
    vertex_sample_feature = torch.cat([
        vertex_sample_feature,      # [B, N, 128] from projection
        vertex_base_feature,        # [B, N, 128] learnable
        vertex_global_feature       # [B, N, 256] from CLS
    ], dim=-1)  # Result: [B, N, 512]

このスクリプトは:
1. vertex_base_feature.bin (学習済み) [N, 128]
2. ゼロ埋めのprojection placeholder [N, 128]
3. ゼロ埋めのglobal placeholder [N, 256]
を連結して [N, 512] を生成します。

実行時にTypeScript側で:
- projection部分 [0:128] を実際の値で上書き
- global部分 [256:512] をID embeddingで上書き
"""

import numpy as np
import argparse
import os
from pathlib import Path


def load_vertex_base_feature(filepath):
    """
    vertex_base_feature.bin をロード
    
    Args:
        filepath: vertex_base_feature.binのパス
        
    Returns:
        numpy.ndarray: [N, 128] の特徴
    """
    if not os.path.exists(filepath):
        print(f"⚠️  {filepath} not found, will use zero-filled features")
        return None
    
    data = np.fromfile(filepath, dtype=np.float32)
    
    # [N, 128] にreshape
    if len(data) % 128 != 0:
        raise ValueError(f"Invalid data size: {len(data)} (expected multiple of 128)")
    
    num_vertices = len(data) // 128
    features = data.reshape(num_vertices, 128)
    
    print(f"✅ Loaded vertex_base_feature: [{num_vertices}, 128]")
    return features


def create_combined_template_features(num_vertices, base_feature=None):
    """
    Template Decoder用の512次元特徴を生成
    
    構造:
        [0:128]   - Projection features (実行時に上書き)
        [128:256] - Base features (学習済みまたはゼロ埋め)
        [256:512] - Global features (実行時に上書き)
    
    Args:
        num_vertices: 頂点数 (通常10595)
        base_feature: Base features [N, 128] (Noneの場合はゼロ埋め)
        
    Returns:
        numpy.ndarray: [N, 512] の連結特徴
    """
    print(f"\n🔧 Creating combined features for {num_vertices} vertices...")
    
    # 1. Projection features placeholder (実行時に上書きされる)
    projection_placeholder = np.zeros((num_vertices, 128), dtype=np.float32)
    print(f"  [0:128]   Projection placeholder: {projection_placeholder.shape}")
    
    # 2. Base features (学習済みまたはゼロ埋め)
    if base_feature is not None and base_feature.shape[0] == num_vertices:
        base_features = base_feature
        print(f"  [128:256] Base features (loaded): {base_features.shape}")
    else:
        base_features = np.zeros((num_vertices, 128), dtype=np.float32)
        print(f"  [128:256] Base features (zero-filled): {base_features.shape}")
    
    # 3. Global features placeholder (実行時に上書きされる)
    global_placeholder = np.zeros((num_vertices, 256), dtype=np.float32)
    print(f"  [256:512] Global placeholder: {global_placeholder.shape}")
    
    # 連結: [N, 128] + [N, 128] + [N, 256] = [N, 512]
    combined = np.concatenate([
        projection_placeholder,  # [0:128]
        base_features,           # [128:256]
        global_placeholder       # [256:512]
    ], axis=1)
    
    print(f"\n✅ Combined features shape: {combined.shape}")
    print(f"   Total size: {combined.nbytes / 1024 / 1024:.2f} MB")
    
    return combined


def save_combined_features(combined, output_path):
    """
    連結済み特徴をバイナリファイルとして保存
    
    Args:
        combined: [N, 512] の特徴
        output_path: 出力ファイルパス
    """
    # Flatten to 1D
    data = combined.flatten()
    
    # Save as float32 binary
    data.tofile(output_path)
    
    print(f"\n💾 Saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"   Shape: [{combined.shape[0]}, 512]")
    print(f"   Data type: float32")


def verify_file(filepath, expected_vertices):
    """
    生成されたファイルを検証
    
    Args:
        filepath: 検証するファイルパス
        expected_vertices: 期待される頂点数
    """
    print(f"\n🔍 Verifying {filepath}...")
    
    data = np.fromfile(filepath, dtype=np.float32)
    expected_size = expected_vertices * 512
    
    if len(data) != expected_size:
        print(f"❌ Size mismatch: {len(data)} != {expected_size}")
        return False
    
    # Reshape
    features = data.reshape(expected_vertices, 512)
    
    # 統計情報
    print(f"✅ File is valid")
    print(f"   Shape: {features.shape}")
    print(f"   Min: {features.min():.6f}")
    print(f"   Max: {features.max():.6f}")
    print(f"   Mean: {features.mean():.6f}")
    print(f"   Non-zeros: {np.count_nonzero(features)} / {features.size}")
    
    # 各セクションの統計
    print(f"\n   Section stats:")
    print(f"     [0:128]   Projection: min={features[:, 0:128].min():.6f}, max={features[:, 0:128].max():.6f}")
    print(f"     [128:256] Base:       min={features[:, 128:256].min():.6f}, max={features[:, 128:256].max():.6f}")
    print(f"     [256:512] Global:     min={features[:, 256:512].min():.6f}, max={features[:, 256:512].max():.6f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare combined template features for GUAVA pipeline'
    )
    parser.add_argument(
        '--base-feature',
        type=str,
        default='vertex_base_feature.bin',
        help='Path to vertex_base_feature.bin (default: vertex_base_feature.bin)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='combined_template_features.bin',
        help='Output file path (default: combined_template_features.bin)'
    )
    parser.add_argument(
        '--num-vertices',
        type=int,
        default=10595,
        help='Number of vertices (default: 10595 for SMPL-X)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎭 GUAVA Template Features Preparation")
    print("=" * 60)
    
    # 1. Load base features (optional)
    base_feature = None
    if os.path.exists(args.base_feature):
        base_feature = load_vertex_base_feature(args.base_feature)
        if base_feature.shape[0] != args.num_vertices:
            print(f"⚠️  Vertex count mismatch: {base_feature.shape[0]} != {args.num_vertices}")
            print(f"   Using zero-filled features instead")
            base_feature = None
    
    # 2. Create combined features
    combined = create_combined_template_features(args.num_vertices, base_feature)
    
    # 3. Save
    save_combined_features(combined, args.output)
    
    # 4. Verify
    verify_file(args.output, args.num_vertices)
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    print(f"\n📝 Next steps:")
    print(f"   1. Copy {args.output} to public/assets/")
    print(f"   2. Update TypeScript to load this file")
    print(f"   3. At runtime, overwrite:")
    print(f"      - [0:128] with projection features from image encoder")
    print(f"      - [256:512] with expanded ID embedding")
    print()


if __name__ == '__main__':
    main()
