"""
GUAVA PLY File Generator (Cloud Assets版)
==========================================
正常動作版 run_guava_gpro_perfect_cloud.py をベースに
PLYファイル生成機能を追加したスクリプト

assetsはクラウドVolume (guava-weights) からマウントするため、
ローカルからの転送による破損問題を回避。

使用方法:
  modal run generate_ply_cloud.py
  modal run generate_ply_cloud.py --output-name my_avatar
"""

import modal
import os
import sys

# --- 1. Volume設定（全てクラウド上の資産を使う） ---
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
weights_volume = modal.Volume.from_name("guava-weights")
ply_output_volume = modal.Volume.from_name("guava-ply-output", create_if_missing=True)

# --- 2. 環境構成（完走実績版を100%維持） ---
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

    # --- 3. ファイル配置 ---
    .add_local_dir("./submodules", remote_path="/root/GUAVA/submodules", copy=True)
    .run_commands(
        "cd /root/GUAVA/submodules/diff-gaussian-rasterization-32 && rm -rf build && pip install . --no-build-isolation",
        "cd /root/GUAVA/submodules/simple-knn && rm -rf build && pip install . --no-build-isolation"
    )
    # PCからはコードだけを送る（assetsは送らない）
    .add_local_dir(".", remote_path="/root/GUAVA", copy=True, ignore=["assets/", "outputs/", ".venv/", ".git/"])
    .run_commands("find /root/GUAVA -maxdepth 3 -name '*.py' | xargs dos2unix")
)

app = modal.App("guava-ply-generator-cloud")


@app.function(
    gpu="L4",
    image=image,
    volumes={
        "/root/EHM_results": ehm_volume,
        "/root/GUAVA/assets": weights_volume,
        "/root/GUAVA/ply_outputs": ply_output_volume
    },
    timeout=3600
)
def generate_ply(
    output_name: str = "guava_avatar",
    save_split: bool = True,
    save_point_cloud: bool = True,
    save_gaussian: bool = True
):
    """
    論文準拠のPLYファイルを生成

    Args:
        output_name: 出力ディレクトリ名
        save_split: SMPLX/UV別々に保存するか
        save_point_cloud: Open3D形式の点群PLYを保存するか
        save_gaussian: 3DGS形式のPLYを保存するか
    """
    import json
    import glob
    import re

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    import torch
    import lightning
    from omegaconf import OmegaConf

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs

    print("=" * 50)
    print("GUAVA PLY Generator (Cloud Assets版)")
    print("=" * 50)

    # --- Assets確認 ---
    print("--- Checking Assets in Cloud Volume ---")
    if os.path.exists("assets"):
        print(f"Assets found: {os.listdir('assets')}")
        if os.path.exists("assets/GUAVA"):
            print(f"GUAVA assets: {os.listdir('assets/GUAVA')}")
    else:
        print("--- [ERROR] Assets folder not found! ---")
        return None

    # --- データパス探索 ---
    # optim_tracking_ehm.pkl が存在するディレクトリを探す
    search_path = "/root/EHM_results/processed_data"
    data_path = None

    def find_tracking_data(base_path, max_depth=3):
        """optim_tracking_ehm.pkl を含むディレクトリを再帰的に探す"""
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

    if os.path.exists(search_path):
        data_path = find_tracking_data(search_path)
        if data_path:
            print(f"自動検出されたデータパス: {data_path}")
            # 確認: 必要なファイルの存在チェック
            tracking_file = os.path.join(data_path, 'optim_tracking_ehm.pkl')
            print(f"トラッキングファイル存在: {os.path.exists(tracking_file)}")
        else:
            print(f"[ERROR] optim_tracking_ehm.pkl が見つかりません")
            print(f"検索パス: {search_path}")
            # デバッグ: ディレクトリ構造を表示
            for root, dirs, files in os.walk(search_path):
                level = root.replace(search_path, '').count(os.sep)
                if level < 3:
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:
                        print(f"{subindent}{file}")
            return None
    else:
        print(f"[ERROR] EHM結果ディレクトリが見つかりません: {search_path}")
        return None

    print(f"データパス: {data_path}")
    print(f"出力名: {output_name}")
    print(f"分割保存: {save_split}")
    print(f"点群PLY: {save_point_cloud}")
    print(f"Gaussian PLY: {save_gaussian}")
    print("=" * 50)

    # --- モデル設定 ---
    model_path = "assets/GUAVA"
    model_config_path = os.path.join(model_path, 'config.yaml')

    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)
    device = 'cuda:0'

    # --- モデル読み込み ---
    print("モデルを読み込み中...")
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    # チェックポイント読み込み
    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best')
    if base_model is None:
        base_model = find_pt_file(ckpt_path, 'latest')

    if base_model is None or not os.path.exists(base_model):
        print(f"[ERROR] チェックポイントが見つかりません: {ckpt_path}")
        if os.path.exists(ckpt_path):
            print(f"checkpoints内容: {os.listdir(ckpt_path)}")
        return None

    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    print(f"モデル読み込み完了: {base_model}")

    # --- データセット設定 ---
    # ConfigDictは通常のdictを返すため、直接設定可能
    meta_cfg['DATASET']['data_path'] = data_path

    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    print(f"データセット読み込み完了: {len(test_dataset)} サンプル")

    # --- 出力ディレクトリ ---
    output_dir = os.path.join("/root/GUAVA/ply_outputs", output_name)
    os.makedirs(output_dir, exist_ok=True)

    video_ids = list(test_dataset.videos_info.keys())
    results = []

    with torch.no_grad():
        for vidx, video_id in enumerate(video_ids):
            print(f"\n処理中: {video_id} [{vidx+1}/{len(video_ids)}]")

            video_output_dir = os.path.join(output_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)

            # ソース情報読み込み
            source_info = test_dataset._load_source_info(video_id)

            # Ubody Gaussians推論
            print("  Ubody Gaussians推論中...")
            import time
            start_time = time.time()
            vertex_gs_dict, up_point_gs_dict, extra_dict = infer_model(source_info)
            infer_time = time.time() - start_time

            # Ubody Gaussian オブジェクト作成
            ubody_gaussians = Ubody_Gaussian(
                meta_cfg.MODEL,
                vertex_gs_dict,
                up_point_gs_dict,
                pruning=True
            )
            ubody_gaussians.init_ehm(infer_model.ehm)
            ubody_gaussians.eval()

            # PLYファイル保存
            ply_files = []

            if save_point_cloud:
                print("  点群PLYを保存中...")
                ubody_gaussians.save_point_ply(video_output_dir, save_split=save_split)
                ply_files.append('canonical.ply')
                if save_split:
                    ply_files.extend(['canonical_smplx.ply', 'canonical_uv.ply'])

            if save_gaussian:
                print("  Gaussian PLYを保存中...")
                ubody_gaussians.save_gaussian_ply(video_output_dir, save_split=save_split)
                ply_files.append('GS_canonical.ply')
                if save_split:
                    ply_files.extend(['GS_canonical_smplx.ply', 'GS_canonical_uv.ply'])

            # 統計情報
            num_template = vertex_gs_dict['positions'].shape[1]
            num_uv = up_point_gs_dict['opacities'].shape[1]
            total = num_template + num_uv

            result = {
                'video_id': video_id,
                'output_dir': video_output_dir,
                'ply_files': ply_files,
                'num_template_gaussians': int(num_template),
                'num_uv_gaussians': int(num_uv),
                'total_gaussians': int(total),
                'inference_time_ms': infer_time * 1000
            }
            results.append(result)

            print(f"  完了: Template={num_template}, UV={num_uv}, Total={total} ({infer_time*1000:.1f}ms)")

    # クリーンアップ
    test_dataset._lmdb_engine.close()

    # サマリー保存
    summary = {
        'data_path': data_path,
        'output_dir': output_dir,
        'total_videos': len(results),
        'settings': {
            'save_split': save_split,
            'save_point_cloud': save_point_cloud,
            'save_gaussian': save_gaussian
        },
        'videos': results
    }

    summary_path = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("生成完了!")
    print("=" * 50)
    print(f"出力ディレクトリ: {output_dir}")
    print(f"処理動画数: {len(results)}")

    # Volume コミット
    ply_output_volume.commit()
    print("--- [SUCCESS] PLY files committed to Volume ---")

    return summary


@app.local_entrypoint()
def main(
    output_name: str = "guava_avatar",
    split: bool = True,
    gaussian_only: bool = False
):
    """
    PLY生成のエントリーポイント

    使用例:
        modal run generate_ply_cloud.py
        modal run generate_ply_cloud.py --output-name my_avatar
        modal run generate_ply_cloud.py --gaussian-only
    """
    print("=== GUAVA PLY Generator (Cloud Assets版) ===")

    result = generate_ply.remote(
        output_name=output_name,
        save_split=split,
        save_point_cloud=not gaussian_only,
        save_gaussian=True
    )

    if result:
        print("\n=== 生成結果 ===")
        print(f"出力ディレクトリ: {result['output_dir']}")
        print(f"処理動画数: {result['total_videos']}")

        for video in result['videos']:
            print(f"\n  {video['video_id']}:")
            print(f"    Template Gaussians: {video['num_template_gaussians']}")
            print(f"    UV Gaussians: {video['num_uv_gaussians']}")
            print(f"    Total: {video['total_gaussians']}")
            print(f"    推論時間: {video['inference_time_ms']:.1f}ms")
            print(f"    PLYファイル:")
            for ply in video['ply_files']:
                print(f"      - {ply}")
    else:
        print("=== [ERROR] PLY生成に失敗しました ===")
