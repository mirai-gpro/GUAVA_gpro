#!/usr/bin/env python3
"""
GUAVA PLY File Generator (Local/Anaconda版)
=============================================
論文「GUAVA: Generalizable Upper Body 3D Gaussian Avatar」に準拠した
PLYファイルをローカル環境で生成するスクリプト

出力PLY形式（論文準拠）:
=========================
GS_canonical.ply - 3D Gaussian Splatting形式
  属性:
    - x, y, z: 位置（canonical空間）
    - nx, ny, nz: 法線（ゼロ初期化）
    - f_dc_0, f_dc_1, f_dc_2: 球面調和係数（RGB→SH変換済み）
    - opacity: 不透明度（inverse sigmoid）
    - scale_0, scale_1, scale_2: スケール（log scale）
    - rot_0, rot_1, rot_2, rot_3: 回転クォータニオン（wxyz順）

canonical.ply - Open3D点群形式
  - 位置 + RGB色

使用方法:
=========
# 基本使用法
python generate_ply_local.py --data_path assets/example/tracked_video/xxx

# 出力先指定
python generate_ply_local.py --data_path <path> --output_dir outputs/ply

# Gaussian PLYのみ
python generate_ply_local.py --data_path <path> --gaussian_only

# 分割保存なし
python generate_ply_local.py --data_path <path> --no_split

環境設定:
=========
conda activate GUAVA
pip install -r requirements.txt
"""

import os
import sys
import argparse
import json
import copy
from pathlib import Path

import torch
import lightning
from omegaconf import OmegaConf
from tqdm import tqdm


def setup_paths():
    """プロジェクトパスをセットアップ"""
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def load_models(meta_cfg, model_path, device):
    """モデルを読み込む"""
    from models.UbodyAvatar import Ubody_Gaussian_inferer, GaussianRenderer
    from utils.general_utils import find_pt_file, calc_parameters

    print("モデルを読み込み中...")

    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device)
    render_model.eval()

    # パラメータ数表示
    _, all_para_num = calc_parameters([infer_model, render_model])
    print(f'パラメータ数: {all_para_num/1000000:.2f}M')

    # チェックポイント読み込み
    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best')
    if base_model is None:
        base_model = find_pt_file(ckpt_path, 'latest')

    if base_model is None or not os.path.exists(base_model):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    render_model.load_state_dict(_state['render_model'], strict=False)

    print(f'モデル読み込み完了: {base_model}')

    return infer_model, render_model


def generate_ply(
    data_path: str,
    output_dir: str = None,
    model_path: str = "assets/GUAVA",
    device: str = "cuda:0",
    save_split: bool = True,
    save_point_cloud: bool = True,
    save_gaussian: bool = True,
    save_deformed: bool = False
):
    """
    論文準拠のPLYファイルを生成

    Args:
        data_path: EHM-Trackerでトラッキング済みのデータパス
        output_dir: 出力ディレクトリ（Noneならdata_pathと同じ場所）
        model_path: GUAVAモデルのパス
        device: 使用デバイス
        save_split: SMPLX/UV別々に保存するか
        save_point_cloud: Open3D形式の点群PLYを保存するか
        save_gaussian: 3DGS形式のPLYを保存するか
        save_deformed: deformed（ポーズ適用後）のPLYも保存するか

    Returns:
        dict: 生成結果のサマリー
    """
    project_root = setup_paths()

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian
    from utils.general_utils import ConfigDict, add_extra_cfgs

    print("=" * 50)
    print("GUAVA PLY Generator (Local)")
    print("=" * 50)
    print(f"データパス: {data_path}")
    print(f"モデルパス: {model_path}")
    print(f"デバイス: {device}")
    print(f"分割保存: {save_split}")
    print(f"点群PLY: {save_point_cloud}")
    print(f"Gaussian PLY: {save_gaussian}")
    print(f"Deformed PLY: {save_deformed}")
    print("=" * 50)

    # 設定読み込み
    model_config_path = os.path.join(model_path, 'config.yaml')
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)

    # モデル読み込み
    infer_model, render_model = load_models(meta_cfg, model_path, device)

    # データセット設定
    OmegaConf.set_readonly(meta_cfg['DATASET'], False)
    meta_cfg['DATASET']['data_path'] = data_path

    test_dataset = TrackedData_infer(
        cfg=meta_cfg,
        split='test',
        device=device,
        test_full=True
    )
    print(f"データセット読み込み完了: {len(test_dataset)} サンプル")

    # 出力ディレクトリ設定
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(data_path), "ply_output")
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
            print("  推論中...")
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
                print("  点群PLY保存中...")
                ubody_gaussians.save_point_ply(video_output_dir, save_split=save_split)
                ply_files.append('canonical.ply')
                if save_split:
                    ply_files.extend(['canonical_smplx.ply', 'canonical_uv.ply'])

            if save_gaussian:
                print("  Gaussian PLY保存中...")
                ubody_gaussians.save_gaussian_ply(video_output_dir, save_split=save_split)
                ply_files.append('GS_canonical.ply')
                if save_split:
                    ply_files.extend(['GS_canonical_smplx.ply', 'GS_canonical_uv.ply'])

            # Deformed PLY保存
            if save_deformed:
                frames = test_dataset.videos_info[video_id]['frames_keys']
                if frames:
                    print("  Deformed PLY保存中...")
                    target_info = test_dataset._load_target_info(video_id, frames[0])
                    deform_gaussian_assets = ubody_gaussians(target_info)
                    ubody_gaussians.save_point_ply(
                        video_output_dir,
                        save_split=False,
                        assets=deform_gaussian_assets
                    )
                    ply_files.append('deformed.ply')

            # 統計情報
            num_template = vertex_gs_dict['positions'].shape[1]
            num_uv = up_point_gs_dict['opacities'].shape[1]

            result = {
                'video_id': video_id,
                'output_dir': video_output_dir,
                'ply_files': ply_files,
                'num_template_gaussians': int(num_template),
                'num_uv_gaussians': int(num_uv),
                'total_gaussians': int(num_template + num_uv),
                'inference_time_ms': infer_time * 1000
            }
            results.append(result)

            print(f"  完了: {num_template + num_uv} Gaussians ({infer_time*1000:.1f}ms)")

    # クリーンアップ
    test_dataset._lmdb_engine.close()

    # サマリー保存
    summary = {
        'data_path': data_path,
        'output_dir': output_dir,
        'model_path': model_path,
        'total_videos': len(results),
        'settings': {
            'save_split': save_split,
            'save_point_cloud': save_point_cloud,
            'save_gaussian': save_gaussian,
            'save_deformed': save_deformed
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
    print(f"サマリー: {summary_path}")

    return summary


def print_ply_format_info():
    """PLY形式の説明を表示"""
    print("""
==========================================================
GUAVA PLY ファイル形式（論文準拠）
==========================================================

【GS_canonical.ply - 3D Gaussian Splatting形式】
  各Gaussianの属性:
    ├── 位置 (float32)
    │   ├── x: X座標（canonical空間）
    │   ├── y: Y座標
    │   └── z: Z座標
    │
    ├── 法線 (float32) ※ゼロで初期化
    │   ├── nx, ny, nz
    │
    ├── 色/SH係数 (float32)
    │   ├── f_dc_0: R成分 (RGB→SH変換済み、係数: 1/0.28209479177387814)
    │   ├── f_dc_1: G成分
    │   └── f_dc_2: B成分
    │
    ├── 不透明度 (float32)
    │   └── opacity: inverse_sigmoid(α) で保存
    │
    ├── スケール (float32)
    │   ├── scale_0: log(sx)
    │   ├── scale_1: log(sy)
    │   └── scale_2: log(sz)
    │
    └── 回転 (float32) - クォータニオン (wxyz順)
        ├── rot_0: w
        ├── rot_1: x
        ├── rot_2: y
        └── rot_3: z

【canonical.ply - Open3D点群形式】
  ├── 位置: x, y, z (float64)
  └── 色: r, g, b (uint8, 0-255)

【構成】
  - Template Gaussians: SMPLX頂点ベース（約10,475点）
  - UV Gaussians: UVマップベース（約140,000点、pruning後）

【対応ソフトウェア】
  - 3D Gaussian Splattingビューア
  - CloudCompare
  - MeshLab
  - Open3D
  - Blender（PLYインポート）
==========================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description='GUAVA PLY Generator - 論文準拠のPLYファイル生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本使用法
  python generate_ply_local.py --data_path assets/example/tracked_video/xxx

  # 出力先を指定
  python generate_ply_local.py --data_path <path> --output_dir outputs/ply

  # Gaussian PLYのみ生成
  python generate_ply_local.py --data_path <path> --gaussian_only

  # 形式の説明を表示
  python generate_ply_local.py --format_info
        """
    )

    parser.add_argument(
        '--data_path', '-d',
        type=str,
        help='EHM-Trackerでトラッキング済みのデータパス'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='出力ディレクトリ（デフォルト: data_path/../ply_output）'
    )
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        default='assets/GUAVA',
        help='GUAVAモデルのパス'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='使用デバイス'
    )
    parser.add_argument(
        '--no_split',
        action='store_true',
        help='SMPLX/UVを分割せずに保存'
    )
    parser.add_argument(
        '--gaussian_only',
        action='store_true',
        help='3DGS形式のPLYのみ生成（点群PLYをスキップ）'
    )
    parser.add_argument(
        '--point_cloud_only',
        action='store_true',
        help='点群PLYのみ生成（3DGS PLYをスキップ）'
    )
    parser.add_argument(
        '--with_deformed',
        action='store_true',
        help='deformed（ポーズ適用後）のPLYも保存'
    )
    parser.add_argument(
        '--format_info',
        action='store_true',
        help='PLY形式の説明を表示して終了'
    )

    args = parser.parse_args()

    if args.format_info:
        print_ply_format_info()
        return

    if args.data_path is None:
        parser.print_help()
        print("\nエラー: --data_path が必要です")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"エラー: データパスが存在しません: {args.data_path}")
        sys.exit(1)

    # 設定
    save_point_cloud = not args.gaussian_only
    save_gaussian = not args.point_cloud_only

    if not save_point_cloud and not save_gaussian:
        print("エラー: --gaussian_only と --point_cloud_only は同時に指定できません")
        sys.exit(1)

    # 実行
    result = generate_ply(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        save_split=not args.no_split,
        save_point_cloud=save_point_cloud,
        save_gaussian=save_gaussian,
        save_deformed=args.with_deformed
    )

    # 結果表示
    print("\n" + "-" * 50)
    print("生成されたファイル:")
    print("-" * 50)
    for video in result['videos']:
        print(f"\n{video['video_id']}:")
        print(f"  Gaussians: {video['total_gaussians']}")
        print(f"    - Template: {video['num_template_gaussians']}")
        print(f"    - UV: {video['num_uv_gaussians']}")
        print(f"  推論時間: {video['inference_time_ms']:.1f}ms")
        print("  ファイル:")
        for ply in video['ply_files']:
            print(f"    - {ply}")


if __name__ == "__main__":
    main()
