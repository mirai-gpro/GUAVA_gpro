"""
Export GUAVA avatar data for mobile runtime
Exports:
1. Gaussian properties (positions, colors, opacity, scale, rotation)
2. SMPLX/FLAME binding data (vertex bindings, face bindings, barycentric coords)
3. Base mesh data for deformation
"""
import modal
import os
import json
import numpy as np

guava_volume = modal.Volume.from_name("guava-results")
weights_volume = modal.Volume.from_name("guava-weights")

# Use the working image from run_guava_gpro_perfect_cloud.py
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "ninja-build",
                 "build-essential", "libglm-dev", "clang", "dos2unix", "ffmpeg",
                 "libsm6", "libxext6", "libxrender-dev")
    .env({"CUDA_HOME": "/usr/local/cuda", "MAX_JOBS": "4",
          "CC": "clang", "CXX": "clang++", "TORCH_CUDA_ARCH_LIST": "8.9"})
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install numpy==1.26.4 scipy"
    )
    .run_commands("pip install chumpy --no-build-isolation")
    .run_commands("pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121")
    .pip_install(
        "lightning", "pytorch-lightning", "omegaconf", "gsplat",
        "opencv-python", "h5py", "tqdm", "scikit-image", "trimesh", "plyfile",
        "lmdb", "lpips", "open3d", "roma", "smplx", "yacs", "ninja",
        "colored", "termcolor", "tabulate", "vispy", "configargparse", "portalocker",
        "fvcore", "iopath", "imageio-ffmpeg"
    )
    .run_commands("pip install numpy==1.26.4")
    .run_commands("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html")
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && rm -rf build && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && rm -rf build && pip install . --no-build-isolation"
    )
    .add_local_dir(".", remote_path="/root/GUAVA", copy=True, ignore=["assets/", "outputs/", ".venv/", ".git/"])
    .run_commands("find /root/GUAVA -maxdepth 3 -name '*.py' | xargs dos2unix")
)

app = modal.App("guava-mobile-export")

@app.function(
    gpu="L4",
    image=image,
    volumes={
        "/root/GUAVA/assets": weights_volume,
        "/root/GUAVA/outputs": guava_volume
    },
    timeout=1800
)
def export_mobile_data():
    """Export avatar data for mobile runtime"""
    import sys
    sys.path.insert(0, "/root/GUAVA")
    os.chdir("/root/GUAVA")

    import torch
    import pickle
    from omegaconf import OmegaConf

    print("=== Starting Mobile Export ===")

    # Load config
    config_path = "assets/GUAVA/config.yaml"
    meta_cfg = OmegaConf.load(config_path)

    # Load checkpoint
    checkpoint_path = "assets/GUAVA/checkpoints/best_160000.pt"
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    # Initialize model
    from models.UbodyAvatar.ubody_gaussian import Ubody_Gaussian_inferer, Ubody_Gaussian

    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL).cuda()
    infer_model.load_state_dict(checkpoint, strict=False)
    infer_model.eval()

    print("Model loaded successfully")

    # Export SMPLX/FLAME base data
    smplx_data = {
        'faces': infer_model.smplx.faces_tensor.cpu().numpy(),
        'v_template': infer_model.v_template.cpu().numpy(),
        'lbs_weights': infer_model.smplx.lbs_weights.cpu().numpy(),
        'faces_uv_idx': infer_model.smplx.faces_uv_idx.cpu().numpy() if hasattr(infer_model.smplx, 'faces_uv_idx') else None,
        'texcoords': infer_model.smplx.texcoords.cpu().numpy() if hasattr(infer_model.smplx, 'texcoords') else None,
    }

    # Save SMPLX data
    smplx_export_path = "/root/GUAVA/outputs/mobile_export/smplx_data.npz"
    os.makedirs(os.path.dirname(smplx_export_path), exist_ok=True)
    np.savez_compressed(smplx_export_path, **{k: v for k, v in smplx_data.items() if v is not None})
    print(f"Saved SMPLX data to {smplx_export_path}")

    # Load existing Gaussian avatar from outputs
    gs_path = "/root/GUAVA/outputs/driving_avatar/render_self_act/driving"

    if os.path.exists(gs_path):
        print(f"Found Gaussian data at {gs_path}")

        # List available files
        files = os.listdir(gs_path)
        print(f"Files: {files}")

        # Load the canonical Gaussian data
        canonical_ply = os.path.join(gs_path, "GS_canonical.ply")
        if os.path.exists(canonical_ply):
            from plyfile import PlyData
            plydata = PlyData.read(canonical_ply)
            vertex = plydata['vertex']

            # Extract Gaussian properties
            gaussian_data = {
                'xyz': np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1),
                'opacity': vertex['opacity'].reshape(-1, 1),
                'scale_0': vertex['scale_0'],
                'scale_1': vertex['scale_1'],
                'scale_2': vertex['scale_2'],
                'rot_0': vertex['rot_0'],
                'rot_1': vertex['rot_1'],
                'rot_2': vertex['rot_2'],
                'rot_3': vertex['rot_3'],
            }

            # Extract SH coefficients
            sh_keys = [k for k in vertex.data.dtype.names if k.startswith('f_dc') or k.startswith('f_rest')]
            for k in sh_keys:
                gaussian_data[k] = vertex[k]

            n_gaussians = gaussian_data['xyz'].shape[0]
            print(f"Loaded {n_gaussians} Gaussians")

            # Save Gaussian data
            gs_export_path = "/root/GUAVA/outputs/mobile_export/gaussians.npz"
            np.savez_compressed(gs_export_path, **gaussian_data)
            print(f"Saved Gaussian data to {gs_export_path}")

    # Export summary
    export_info = {
        'n_gaussians': n_gaussians,
        'n_vertices': smplx_data['v_template'].shape[0] if smplx_data['v_template'] is not None else 0,
        'n_faces': smplx_data['faces'].shape[0] if smplx_data['faces'] is not None else 0,
        'files': [
            'smplx_data.npz',
            'gaussians.npz',
        ]
    }

    info_path = "/root/GUAVA/outputs/mobile_export/export_info.json"
    with open(info_path, 'w') as f:
        json.dump(export_info, f, indent=2)

    print("\n=== Export Summary ===")
    print(f"Gaussians: {export_info['n_gaussians']}")
    print(f"Vertices: {export_info['n_vertices']}")
    print(f"Faces: {export_info['n_faces']}")
    print(f"Files exported to /root/GUAVA/outputs/mobile_export/")

    # Commit to volume
    guava_volume.commit()

    return export_info

@app.local_entrypoint()
def main():
    result = export_mobile_data.remote()
    print("\n=== Export Complete ===")
    print(result)
