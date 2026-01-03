"""
Reduce Gaussian count for Medium tier rendering
Input: GS_canonical.ply (full quality ~50K+ Gaussians)
Output: GS_medium.ply (~10K Gaussians)
"""
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os


def load_ply(path):
    """Load 3DGS PLY file"""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    # Extract all attributes
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Count Gaussians
    n_gaussians = xyz.shape[0]
    print(f"Loaded {n_gaussians:,} Gaussians from {path}")

    return plydata, n_gaussians


def reduce_gaussians(plydata, target_count, method='importance'):
    """
    Reduce Gaussian count using various methods

    Methods:
    - 'random': Random sampling
    - 'importance': Keep Gaussians with higher opacity
    - 'spatial': Spatial voxel-based reduction
    """
    vertex = plydata['vertex']
    n_original = len(vertex.data)

    if target_count >= n_original:
        print(f"Target {target_count} >= original {n_original}, no reduction needed")
        return plydata

    if method == 'random':
        # Random sampling
        indices = np.random.choice(n_original, target_count, replace=False)
        indices = np.sort(indices)

    elif method == 'importance':
        # Keep Gaussians with higher opacity
        # opacity is stored as sigmoid inverse, so we need to convert
        opacity = vertex['opacity']
        # Higher opacity = more important
        indices = np.argsort(opacity)[-target_count:]
        indices = np.sort(indices)

    elif method == 'spatial':
        # Voxel-based spatial reduction
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

        # Determine voxel size to get approximately target_count
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        bbox_size = bbox_max - bbox_min

        # Estimate voxel size
        volume = np.prod(bbox_size)
        voxel_volume = volume / target_count
        voxel_size = voxel_volume ** (1/3)

        # Assign voxel indices
        voxel_indices = ((xyz - bbox_min) / voxel_size).astype(int)
        voxel_keys = voxel_indices[:, 0] * 10000000 + voxel_indices[:, 1] * 10000 + voxel_indices[:, 2]

        # Keep one Gaussian per voxel (highest opacity)
        opacity = vertex['opacity']
        selected = []
        seen_voxels = {}

        for i in np.argsort(-opacity):  # Sort by opacity descending
            key = voxel_keys[i]
            if key not in seen_voxels:
                seen_voxels[key] = i
                selected.append(i)
                if len(selected) >= target_count:
                    break

        indices = np.sort(selected)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Create new vertex data
    new_data = vertex.data[indices]
    new_vertex = PlyElement.describe(new_data, 'vertex')

    print(f"Reduced from {n_original:,} to {len(indices):,} Gaussians ({method} method)")

    return PlyData([new_vertex])


def save_ply(plydata, path):
    """Save PLY file"""
    plydata.write(path)
    size = os.path.getsize(path)
    print(f"Saved to {path} ({size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description='Reduce Gaussian count for mobile rendering')
    parser.add_argument('--input', '-i', default='assets/GS_canonical.ply', help='Input PLY file')
    parser.add_argument('--output', '-o', default='assets/GS_medium.ply', help='Output PLY file')
    parser.add_argument('--target', '-t', type=int, default=10000, help='Target Gaussian count')
    parser.add_argument('--method', '-m', choices=['random', 'importance', 'spatial'],
                        default='importance', help='Reduction method')
    args = parser.parse_args()

    # Load
    plydata, n_original = load_ply(args.input)

    # Reduce
    reduced = reduce_gaussians(plydata, args.target, args.method)

    # Save
    save_ply(reduced, args.output)

    # Summary
    reduction_ratio = args.target / n_original * 100
    print(f"\nSummary:")
    print(f"  Original: {n_original:,} Gaussians")
    print(f"  Reduced:  {args.target:,} Gaussians ({reduction_ratio:.1f}%)")


if __name__ == '__main__':
    main()
