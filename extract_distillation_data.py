"""
extract_distillation_data.py - GUAVA蒸留データ抽出スクリプト (v2)
==============================================================

Geminiの分析を反映:
- render_modelを独立してロード
- 明示的にレンダリングループを回してRefinerを発動
- TrackedData_inferからカメラ情報を取得

使用方法:
  modal run extract_distillation_data.py
  modal run extract_distillation_data.py --num-frames 1000
"""

import modal
import os
import sys

# --- Volume設定 ---
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
weights_volume = modal.Volume.from_name("guava-weights")
distill_volume = modal.Volume.from_name("guava-distillation-data", create_if_missing=True)

# --- 環境構成 (generate_ply_cloud.pyと同じ) ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "git", "ninja-build",
        "build-essential", "libglm-dev", "clang", "dos2unix", "ffmpeg",
        "libsm6", "libxext6", "libxrender-dev"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "MAX_JOBS": "4", "CC": "clang", "CXX": "clang++", "TORCH_CUDA_ARCH_LIST": "8.9"
    })
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install \"numpy==1.26.4\" \"scipy\""
    )
    .run_commands("pip install chumpy --no-build-isolation")
    .run_commands("pip install \"torch==2.1.0\" \"torchvision==0.16.0\" --extra-index-url https://download.pytorch.org/whl/cu121")
    .pip_install(
        "lightning", "pytorch-lightning", "omegaconf", "gsplat",
        "opencv-python", "h5py", "tqdm", "scikit-image", "trimesh", "plyfile",
        "lmdb", "lpips", "open3d", "roma", "smplx", "yacs", "ninja",
        "colored", "termcolor", "tabulate", "vispy", "configargparse", "portalocker",
        "fvcore", "iopath", "imageio-ffmpeg"
    )
    .run_commands("pip install \"numpy==1.26.4\"")
    .run_commands("pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html")
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && rm -rf build && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && rm -rf build && pip install . --no-build-isolation"
    )
    .add_local_dir(".", remote_path="/root/GUAVA", copy=True, ignore=["assets/", "outputs/", ".venv/", ".git/"])
    .run_commands("find /root/GUAVA -maxdepth 3 -name '*.py' | xargs dos2unix")
)

app = modal.App("guava-distillation-extractor-v2")


def find_tracking_data(base_path, max_depth=3):
    """optim_tracking_ehm.pklを探す"""
    if max_depth <= 0:
        return None
    tracking_file = os.path.join(base_path, 'optim_tracking_ehm.pkl')
    if os.path.exists(tracking_file):
        return base_path
    if os.path.isdir(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                result = find_tracking_data(item_path, max_depth - 1)
                if result:
                    return result
    return None


@app.function(
    gpu="L4",
    image=image,
    volumes={
        "/root/EHM_results": ehm_volume,
        "/root/GUAVA/assets": weights_volume,
        "/root/distill_data": distill_volume
    },
    timeout=7200
)
def extract_distillation_data(
    output_name: str = "distill_dataset",
    num_frames: int = 1000,
    num_camera_angles: int = 5,  # カメラアングル数
):
    """
    GUAVA推論+レンダリングを実行して蒸留用データを抽出
    複数カメラアングルでデータを増やす
    """
    import json
    import numpy as np
    from tqdm import tqdm
    import math
    
    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")
    
    import torch
    import lightning
    from omegaconf import OmegaConf
    
    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian
    from models.UbodyAvatar.gaussian_render import GaussianRenderer
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs
    
    print("=" * 70)
    print("🔬 GUAVA Distillation Data Extractor v3 (Multi-Angle)")
    print("=" * 70)
    print(f"📊 目標フレーム数: {num_frames}")
    print(f"📊 カメラアングル数: {num_camera_angles}")
    
    # --- カメラアングル生成関数 ---
    def generate_camera_angles(num_angles):
        """
        複数のカメラアングルを生成
        正面 + 左右回転 + 上下回転
        """
        angles = []
        
        # 正面
        angles.append({'yaw': 0, 'pitch': 0, 'name': 'front'})
        
        if num_angles >= 3:
            # 左右
            angles.append({'yaw': 15, 'pitch': 0, 'name': 'left_15'})
            angles.append({'yaw': -15, 'pitch': 0, 'name': 'right_15'})
        
        if num_angles >= 5:
            # 上下
            angles.append({'yaw': 0, 'pitch': 10, 'name': 'up_10'})
            angles.append({'yaw': 0, 'pitch': -10, 'name': 'down_10'})
        
        if num_angles >= 9:
            # 斜め
            angles.append({'yaw': 10, 'pitch': 5, 'name': 'left_up'})
            angles.append({'yaw': -10, 'pitch': 5, 'name': 'right_up'})
            angles.append({'yaw': 10, 'pitch': -5, 'name': 'left_down'})
            angles.append({'yaw': -10, 'pitch': -5, 'name': 'right_down'})
        
        if num_angles >= 13:
            # より大きな角度
            angles.append({'yaw': 25, 'pitch': 0, 'name': 'left_25'})
            angles.append({'yaw': -25, 'pitch': 0, 'name': 'right_25'})
            angles.append({'yaw': 0, 'pitch': 15, 'name': 'up_15'})
            angles.append({'yaw': 0, 'pitch': -15, 'name': 'down_15'})
        
        return angles[:num_angles]
    
    def create_rotation_matrix(yaw_deg, pitch_deg, device):
        """
        Yaw (左右回転) と Pitch (上下回転) から回転行列を生成
        """
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        
        # Yaw (Y軸回転)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Ry = torch.tensor([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype=torch.float32, device=device)
        
        # Pitch (X軸回転)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cp, -sp],
            [0, sp, cp]
        ], dtype=torch.float32, device=device)
        
        # 合成
        R = Ry @ Rx
        return R
    
    def apply_camera_rotation(base_w2c, yaw_deg, pitch_deg, device):
        """
        基本のworld-to-camera行列に回転を適用
        """
        if base_w2c.dim() == 2:
            base_w2c = base_w2c.unsqueeze(0)
        
        batch_size = base_w2c.shape[0]
        R_delta = create_rotation_matrix(yaw_deg, pitch_deg, device)
        
        # 4x4行列に拡張
        R_delta_4x4 = torch.eye(4, device=device)
        R_delta_4x4[:3, :3] = R_delta
        
        # 回転を適用
        new_w2c = R_delta_4x4.unsqueeze(0) @ base_w2c
        
        return new_w2c
    
    # --- Assets確認 ---
    print("\n--- Checking Assets ---")
    if not (os.path.exists("assets") and os.path.exists("assets/GUAVA")):
        print("❌ Assets folder not found!")
        return None
    print(f"✅ GUAVA assets: {os.listdir('assets/GUAVA')}")
    
    # --- データパス探索 ---
    search_path = "/root/EHM_results/processed_data"
    data_path = find_tracking_data(search_path) if os.path.exists(search_path) else None
    
    if not data_path:
        print(f"❌ Tracking data not found in {search_path}")
        return None
    print(f"✅ データパス: {data_path}")
    
    # --- モデル設定 ---
    model_path = "assets/GUAVA"
    model_config_path = os.path.join(model_path, 'config.yaml')
    
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)
    
    lightning.fabric.seed_everything(10)
    device = 'cuda:0'
    
    # --- チェックポイントロード ---
    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best') or find_pt_file(ckpt_path, 'latest')
    
    if not base_model or not os.path.exists(base_model):
        print(f"❌ チェックポイントが見つかりません: {ckpt_path}")
        return None
    
    print(f"\n🔄 チェックポイント読み込み: {base_model}")
    state = torch.load(base_model, map_location='cpu', weights_only=True)
    
    # --- Infer Model読み込み ---
    print("\n🔄 Infer Model読み込み中...")
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()
    infer_model.load_state_dict(state['model'], strict=False)
    print(f"✅ Infer Model読み込み完了")
    
    # --- Render Model読み込み ---
    print("\n🔄 Render Model読み込み中...")
    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    render_model.eval()
    
    # Neural Refinerの重みをロード
    if 'render_model' in state:
        print(f"  Found 'render_model' in checkpoint")
        render_model.load_state_dict(state['render_model'], strict=False)
        print(f"✅ Render Model weights loaded from state['render_model']")
    else:
        print("⚠️ No render_model found in checkpoint")
    
    print(f"✅ Render Model読み込み完了")
    
    # --- Hook設定 ---
    captured_data = {"input": None, "output": None}
    
    def refiner_hook(module, input, output):
        captured_data["input"] = input[0].detach().cpu()
        captured_data["output"] = output.detach().cpu()
    
    render_model.nerual_refiner.register_forward_hook(refiner_hook)
    print("✅ Hook registered on neural_refiner")
    
    # --- 出力ディレクトリ ---
    output_dir = os.path.join("/root/distill_data", output_name)
    os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
    
    # --- データセット設定 ---
    meta_cfg['DATASET']['data_path'] = data_path
    OmegaConf.set_readonly(meta_cfg._dot_config, False)
    meta_cfg._dot_config.DATASET.data_path = data_path
    OmegaConf.set_readonly(meta_cfg._dot_config, True)
    
    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    print(f"✅ データセット読み込み完了: {len(test_dataset)} サンプル")
    
    video_ids = list(test_dataset.videos_info.keys())
    print(f"  video_ids: {video_ids}")
    
    # --- カメラアングル生成 ---
    camera_angles = generate_camera_angles(num_camera_angles)
    print(f"\n📷 カメラアングル: {[a['name'] for a in camera_angles]}")
    
    # --- データ抽出ループ ---
    print("\n" + "=" * 70)
    print("🚀 データ抽出開始 (Multi-Angle)")
    print("=" * 70)
    
    saved_count = 0
    
    with torch.no_grad():
        for vidx, video_id in enumerate(video_ids):
            if saved_count >= num_frames:
                print(f"\n✅ 目標フレーム数 ({num_frames}) に到達")
                break
            
            print(f"\n--- 動画: {video_id} [{vidx+1}/{len(video_ids)}] ---")
            
            # ソース情報読み込み
            source_info = test_dataset._load_source_info(video_id)
            
            # Gaussian推論
            print("  🔄 Gaussian推論中...")
            vertex_gs_dict, up_point_gs_dict, extra_dict = infer_model(source_info)
            print(f"  ✅ Gaussian推論完了")
            
            # Ubody Gaussian作成
            ubody_gaussians = Ubody_Gaussian(
                meta_cfg.MODEL,
                vertex_gs_dict,
                up_point_gs_dict,
                pruning=True
            )
            ubody_gaussians.init_ehm(infer_model.ehm)
            ubody_gaussians.eval()
            
            # Canonical Gaussiansを取得
            if not ubody_gaussians._canoical:
                ubody_gaussians.get_canoical_gaussians()
            print(f"  ✅ Canonical Gaussians取得完了")
            
            # Gaussian assetsを構築 (全フレーム共通)
            xyz_all = torch.cat([ubody_gaussians._smplx_xyz, ubody_gaussians._uv_xyz_cano], dim=1)
            opacity_all = torch.cat([ubody_gaussians._smplx_opacity, ubody_gaussians._uv_opacity_cano], dim=1)
            scaling_all = torch.cat([ubody_gaussians._smplx_scaling, ubody_gaussians._uv_scaling_cano], dim=1)
            rotation_all = torch.cat([ubody_gaussians._smplx_rotation, ubody_gaussians._uv_rotation_cano], dim=1)
            features_all = torch.cat([ubody_gaussians._smplx_features_color, ubody_gaussians._uv_features_color], dim=1)
            
            gaussian_assets = {
                'xyz': xyz_all.to(device),
                'opacity': opacity_all.to(device),
                'scaling': scaling_all.to(device),
                'rotation': rotation_all.to(device),
                'features_color': features_all.to(device),
            }
            
            dataset_len = len(test_dataset)
            total_iterations = dataset_len * len(camera_angles)
            
            print(f"  📊 フレーム数: {dataset_len}, アングル数: {len(camera_angles)}, 合計: {total_iterations}")
            
            # フレームをイテレート
            pbar = tqdm(range(dataset_len), desc=f"  Rendering")
            for idx in pbar:
                if saved_count >= num_frames:
                    break
                
                try:
                    # データセットからバッチを取得
                    batch = test_dataset[idx]
                    
                    if batch is None:
                        continue
                    
                    # ベースのカメラ行列を取得
                    if hasattr(batch, 'keys') and 'w2c_cam' in batch.get('source', {}):
                        base_w2c = batch['source']['w2c_cam']
                    elif hasattr(batch, 'keys') and 'target' in batch and 'w2c_cam' in batch['target']:
                        base_w2c = batch['target']['w2c_cam']
                    else:
                        base_w2c = torch.eye(4, device=device)
                    
                    if base_w2c.dim() == 2:
                        base_w2c = base_w2c.unsqueeze(0)
                    base_w2c = base_w2c.to(device)
                    
                    # 各カメラアングルでレンダリング
                    for angle in camera_angles:
                        if saved_count >= num_frames:
                            break
                        
                        # カメラ回転を適用
                        w2c = apply_camera_rotation(base_w2c, angle['yaw'], angle['pitch'], device)
                        
                        cam_params = {
                            'image_height': torch.tensor([512], device=device),
                            'image_width': torch.tensor([512], device=device),
                            'tanfovx': torch.tensor([meta_cfg.MODEL.invtanfov], device=device),
                            'tanfovy': torch.tensor([meta_cfg.MODEL.invtanfov], device=device),
                            'world_view_transform': w2c,
                            'full_proj_transform': w2c,
                            'camera_center': torch.zeros(1, 3, device=device),
                        }
                        
                        # レンダリング実行
                        render_output = render_model(
                            gaussian_assets,
                            cam_params,
                            bg=1.0
                        )
                        
                        # Hookでキャプチャされたデータを保存
                        if captured_data["input"] is not None and captured_data["output"] is not None:
                            # 不正なサンプルをスキップ
                            input_data = captured_data["input"][0]
                            
                            # 分散チェック (背景のみのサンプルをスキップ)
                            if input_data.std() < 0.01:
                                print(f"    ⚠️ Skipping sample (low variance: {input_data.std():.4f})")
                                captured_data["input"] = None
                                captured_data["output"] = None
                                continue
                            
                            # NaNチェック
                            if torch.isnan(input_data).any():
                                print(f"    ⚠️ Skipping sample (contains NaN)")
                                captured_data["input"] = None
                                captured_data["output"] = None
                                continue
                            
                            save_id = f"{saved_count:06d}"
                            
                            # 32ch特徴マップを保存
                            feat_path = os.path.join(output_dir, 'features', f'{save_id}.pt')
                            torch.save(input_data.clone(), feat_path)
                            
                            # RGB出力を保存
                            rgb_path = os.path.join(output_dir, 'rgb', f'{save_id}.pt')
                            torch.save(captured_data["output"][0].clone(), rgb_path)
                            
                            saved_count += 1
                            
                            # リセット
                            captured_data["input"] = None
                            captured_data["output"] = None
                        
                        pbar.set_postfix({'saved': saved_count, 'angle': angle['name']})
                    
                except Exception as e:
                    if idx < 3:
                        import traceback
                        print(f"    ❌ Frame {idx} error: {type(e).__name__}: {e}")
                        traceback.print_exc()
                    continue
            
            print(f"\n  📊 保存済み: {saved_count} ペア")
    
    # クリーンアップ
    if hasattr(test_dataset, '_lmdb_engine') and test_dataset._lmdb_engine:
        test_dataset._lmdb_engine.close()
    
    # メタデータ保存
    metadata = {
        'num_samples': saved_count,
        'feature_shape': [32, 512, 512],
        'rgb_shape': [3, 512, 512],
        'source': 'GUAVA Neural Refiner',
        'teacher_model': 'StyleUNet (107MB)',
        'camera_angles': [a['name'] for a in camera_angles],
        'num_camera_angles': len(camera_angles)
    }
    torch.save(metadata, os.path.join(output_dir, 'metadata.pt'))
    
    # 統計保存
    stats = {
        'total_frames': saved_count,
        'saved_pairs': saved_count,
        'save_dir': output_dir,
        'video_ids': video_ids,
        'camera_angles': [a['name'] for a in camera_angles]
    }
    with open(os.path.join(output_dir, 'extraction_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 データ抽出完了!")
    print("=" * 70)
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"📊 抽出フレーム数: {saved_count}")
    print(f"📷 カメラアングル: {[a['name'] for a in camera_angles]}")
    
    # Volume コミット
    distill_volume.commit()
    print("\n✅ Data committed to Volume")
    
    return stats


@app.function(
    image=image,
    volumes={"/root/distill_data": distill_volume},
)
def verify_extracted_data(dataset_name: str = "distill_dataset"):
    """抽出したデータを検証"""
    import torch
    
    data_dir = f"/root/distill_data/{dataset_name}"
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found: {data_dir}")
        return None
    
    features_dir = os.path.join(data_dir, 'features')
    rgb_dir = os.path.join(data_dir, 'rgb')
    
    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.pt')])
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.pt')])
    
    print(f"📊 Features: {len(feature_files)} files")
    print(f"📊 RGB: {len(rgb_files)} files")
    
    if len(feature_files) == 0:
        print("❌ No data found")
        return None
    
    # サンプル検証
    print("\n--- Sample Verification ---")
    for i in range(min(3, len(feature_files))):
        feat = torch.load(os.path.join(features_dir, feature_files[i]))
        rgb = torch.load(os.path.join(rgb_dir, rgb_files[i]))
        
        print(f"\nSample {i}:")
        print(f"  Feature shape: {feat.shape}")
        print(f"  Feature range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    
    return {
        'num_samples': len(feature_files),
        'feature_shape': list(feat.shape),
        'rgb_shape': list(rgb.shape)
    }


@app.local_entrypoint()
def main(
    action: str = "extract",
    output_name: str = "distill_dataset",
    num_frames: int = 1000,
    num_angles: int = 5,
):
    """
    エントリポイント
    
    使用方法:
        modal run extract_distillation_data.py --action extract --num-frames 1000 --num-angles 5
        modal run extract_distillation_data.py --action verify
    """
    
    if action == "extract":
        print("🚀 Starting distillation data extraction...")
        print(f"📷 Camera angles: {num_angles}")
        result = extract_distillation_data.remote(
            output_name=output_name,
            num_frames=num_frames,
            num_camera_angles=num_angles,
        )
        print(f"\n📊 Result: {result}")
        
    elif action == "verify":
        print("🔍 Verifying extracted data...")
        result = verify_extracted_data.remote(output_name)
        print(f"\n📊 Result: {result}")
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: extract, verify")
