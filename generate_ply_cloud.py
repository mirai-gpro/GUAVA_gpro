"""
GUAVA PLY File Generator - 三角形データ追加版
==========================================
公式save_gaussian_ply()の出力に、EHMメッシュの三角形データを追加

使用方法:
  modal run generate_ply_cloud.py
  modal run generate_ply_cloud.py --output-name my_avatar
"""

import modal
import os
import sys

# --- 1. Volume設定 ---
ehm_volume = modal.Volume.from_name("ehm-tracker-output")
weights_volume = modal.Volume.from_name("guava-weights")
ply_output_volume = modal.Volume.from_name("guava-ply-output", create_if_missing=True)

# --- 2. 環境構成 ---
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

app = modal.App("guava-ply-generator-with-faces")


def add_faces_to_ply(source_ply_path, output_ply_path, faces):
    """
    既存のPLYファイルに三角形データを追加
    
    Args:
        source_ply_path: 元のPLYファイル (頂点のみ)
        output_ply_path: 出力PLYファイル (頂点+三角形)
        faces: 三角形データ (numpy array, shape: [num_faces, 3])
    """
    import struct
    
    # Step 1: 元のPLYファイルを読み込み
    with open(source_ply_path, 'rb') as f:
        header_bytes = f.read(10000)
        header_text = header_bytes.decode('ascii', errors='ignore')
        end_header_pos = header_text.find('end_header')
        
        if end_header_pos == -1:
            raise ValueError("Invalid PLY file: no end_header found")
        
        # ヘッダー解析
        header_lines = header_text[:end_header_pos].split('\n')
        vertex_count = 0
        properties = []
        
        for line in header_lines:
            line = line.strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(line)  # プロパティ行全体を保存
        
        # 頂点データ読み込み
        header_size = end_header_pos + len('end_header\n')
        f.seek(header_size)
        vertex_data = f.read()
    
    # Step 2: 新しいヘッダーを作成（三角形データ追加）
    num_faces = faces.shape[0]
    
    new_header = "ply\n"
    new_header += "format binary_little_endian 1.0\n"
    new_header += f"element vertex {vertex_count}\n"
    
    # プロパティ一覧を追加
    for prop in properties:
        new_header += prop + "\n"
    
    # 三角形要素を追加
    new_header += f"element face {num_faces}\n"
    new_header += "property list uchar uint vertex_indices\n"
    new_header += "end_header\n"
    
    # Step 3: 新しいPLYファイルを保存
    with open(output_ply_path, 'wb') as f:
        # ヘッダー書き込み
        f.write(new_header.encode('ascii'))
        
        # 頂点データ書き込み（既存データをそのまま）
        f.write(vertex_data)
        
        # 三角形データ書き込み
        for i in range(num_faces):
            f.write(struct.pack('<B', 3))  # 3頂点の三角形
            f.write(struct.pack('<III', 
                int(faces[i, 0]), 
                int(faces[i, 1]), 
                int(faces[i, 2])
            ))
    
    return output_ply_path


def save_web_compatible_ply(ubody_gaussians, output_dir, faces=None):
    """
    gvrm.ts/ply.ts互換のPLYファイルを保存

    フォーマット (全てfloat32):
    - x, y, z (位置)
    - nx, ny, nz (法線)
    - f_dc_0, f_dc_1, f_dc_2 (SH形式の色)
    - scale_0, scale_1, scale_2 (スケール)

    オプションで三角形データも追加
    """
    import numpy as np
    import struct
    import os

    # Canonical Gaussiansを取得
    if not ubody_gaussians._canoical:
        ubody_gaussians.get_canoical_gaussians()

    # データ取得
    xyz = ubody_gaussians._xyz.detach().cpu().numpy()  # [N, 3]
    features_dc = ubody_gaussians._features_dc.detach().cpu().numpy()  # [N, 1, 3]
    scaling = ubody_gaussians._scaling.detach().cpu().numpy()  # [N, 3]

    num_points = xyz.shape[0]

    # 法線（ゼロで初期化 - 後でメッシュから計算可能）
    normals = np.zeros((num_points, 3), dtype=np.float32)

    # SH係数を取り出し
    f_dc = features_dc[:, 0, :]  # [N, 3]

    # PLYヘッダー作成
    output_path = os.path.join(output_dir, 'avatar_web.ply')

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {num_points}\n"
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"
    header += "property float nx\n"
    header += "property float ny\n"
    header += "property float nz\n"
    header += "property float f_dc_0\n"
    header += "property float f_dc_1\n"
    header += "property float f_dc_2\n"
    header += "property float scale_0\n"
    header += "property float scale_1\n"
    header += "property float scale_2\n"

    # 三角形データがあれば追加
    if faces is not None:
        num_faces = faces.shape[0]
        header += f"element face {num_faces}\n"
        header += "property list uchar uint vertex_indices\n"

    header += "end_header\n"

    # バイナリデータ作成
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))

        # 頂点データ書き込み
        for i in range(num_points):
            # x, y, z
            f.write(struct.pack('<fff', xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            # nx, ny, nz
            f.write(struct.pack('<fff', normals[i, 0], normals[i, 1], normals[i, 2]))
            # f_dc_0, f_dc_1, f_dc_2
            f.write(struct.pack('<fff', f_dc[i, 0], f_dc[i, 1], f_dc[i, 2]))
            # scale_0, scale_1, scale_2
            f.write(struct.pack('<fff', scaling[i, 0], scaling[i, 1], scaling[i, 2]))

        # 三角形データ書き込み
        if faces is not None:
            for i in range(num_faces):
                f.write(struct.pack('<B', 3))
                f.write(struct.pack('<III',
                    int(faces[i, 0]),
                    int(faces[i, 1]),
                    int(faces[i, 2])
                ))

    return output_path, num_points


def verify_ply_format(ply_path):
    """PLYファイルの形式を検証"""
    import struct
    
    try:
        with open(ply_path, 'rb') as f:
            header_bytes = f.read(10000)
            header_text = header_bytes.decode('ascii', errors='ignore')
            end_header_pos = header_text.find('end_header')
            
            if end_header_pos == -1:
                return {"error": "Invalid PLY: no end_header"}
            
            header_lines = header_text[:end_header_pos].split('\n')
            
            vertex_count = 0
            face_count = 0
            properties = []
            in_vertex_section = False
            
            for line in header_lines:
                line = line.strip()
                
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                    in_vertex_section = True
                elif line.startswith('element face'):
                    face_count = int(line.split()[2])
                    in_vertex_section = False
                elif line.startswith('property') and in_vertex_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        properties.append(parts[-1])
            
            file_size = os.path.getsize(ply_path)
            header_size = end_header_pos + len('end_header\n')
            vertex_data_size = vertex_count * len(properties) * 4
            face_data_size = face_count * (1 + 3 * 4) if face_count > 0 else 0
            expected_size = header_size + vertex_data_size + face_data_size
            actual_data_size = file_size - header_size
            
            return {
                "vertex_count": vertex_count,
                "face_count": face_count,
                "property_count": len(properties),
                "properties": properties,
                "file_size_mb": file_size / (1024 * 1024),
                "header_size": header_size,
                "expected_data_size": vertex_data_size + face_data_size,
                "actual_data_size": actual_data_size,
                "size_match": abs(actual_data_size - (vertex_data_size + face_data_size)) < 1000,
                "has_faces": face_count > 0,
                "has_opacity": 'opacity' in properties,
                "has_rotation": 'rot_0' in properties,
                "has_scale": 'scale_0' in properties,
            }
    except Exception as e:
        return {"error": str(e)}


def extract_ply_sample(ply_path, num_samples=5):
    """PLYファイルから最初の数頂点のデータを抽出"""
    import struct
    
    try:
        with open(ply_path, 'rb') as f:
            header_bytes = f.read(10000)
            header_text = header_bytes.decode('ascii', errors='ignore')
            end_header_pos = header_text.find('end_header')
            
            header_lines = header_text[:end_header_pos].split('\n')
            
            vertex_count = 0
            properties = []
            in_vertex_section = False
            
            for line in header_lines:
                line = line.strip()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                    in_vertex_section = True
                elif line.startswith('element face'):
                    in_vertex_section = False
                elif line.startswith('property') and in_vertex_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        properties.append(parts[-1])
            
            header_size = end_header_pos + len('end_header\n')
            f.seek(header_size)
            
            stride = len(properties) * 4
            
            print(f"\n    📊 First {num_samples} vertices:")
            for i in range(min(num_samples, vertex_count)):
                vertex_data = f.read(stride)
                values = struct.unpack('<' + 'f' * len(properties), vertex_data)
                
                print(f"      Vertex {i}:")
                for j, prop in enumerate(properties[:12]):
                    print(f"        {prop:12s} = {values[j]:10.6f}")
                
                if i < num_samples - 1:
                    print()
                
    except Exception as e:
        print(f"    ❌ Sample extraction failed: {e}")


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
    save_split: bool = False,
    save_point_cloud: bool = False,
    save_gaussian: bool = True,
    save_web: bool = True,
    verify_format: bool = True,
    extract_samples: bool = True
):
    """
    論文準拠の完全なPLYファイルを生成（三角形データ付き）

    Args:
        output_name: 出力ディレクトリ名
        save_split: SMPLX/UV別々に保存するか
        save_point_cloud: Open3D形式の点群PLYを保存するか
        save_gaussian: 3DGS形式のPLYを保存するか（論文準拠）
        save_web: gvrm.ts/ply.ts互換のPLYを保存するか
        verify_format: PLYファイル形式を検証するか
        extract_samples: サンプルデータを抽出するか
    """
    import json
    import numpy as np

    os.chdir("/root/GUAVA")
    sys.path.insert(0, "/root/GUAVA")

    import torch
    import lightning
    from omegaconf import OmegaConf

    from dataset import TrackedData_infer
    from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian
    from utils.general_utils import ConfigDict, find_pt_file, add_extra_cfgs

    print("=" * 70)
    print("GUAVA PLY Generator (三角形データ追加版)")
    print("公式PLY + EHMメッシュの三角形データ")
    print("=" * 70)

    # --- Assets確認 ---
    print("\n--- Checking Assets ---")
    if os.path.exists("assets") and os.path.exists("assets/GUAVA"):
        print(f"✅ GUAVA assets: {os.listdir('assets/GUAVA')}")
    else:
        print("❌ Assets folder not found!")
        return None

    # --- データパス探索 ---
    search_path = "/root/EHM_results/processed_data"
    
    def find_tracking_data(base_path, max_depth=3):
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

    data_path = None
    if os.path.exists(search_path):
        data_path = find_tracking_data(search_path)
        if data_path:
            print(f"✅ データパス: {data_path}")
        else:
            print(f"❌ optim_tracking_ehm.pkl が見つかりません")
            return None
    else:
        print(f"❌ EHM結果ディレクトリが見つかりません: {search_path}")
        return None

    print(f"\n📝 出力名: {output_name}")
    print("=" * 70)

    # --- モデル設定 ---
    model_path = "assets/GUAVA"
    model_config_path = os.path.join(model_path, 'config.yaml')

    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    lightning.fabric.seed_everything(10)
    device = 'cuda:0'

    # --- モデル読み込み ---
    print("\n🔄 モデルを読み込み中...")
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    ckpt_path = os.path.join(model_path, 'checkpoints')
    base_model = find_pt_file(ckpt_path, 'best')
    if base_model is None:
        base_model = find_pt_file(ckpt_path, 'latest')

    if base_model is None or not os.path.exists(base_model):
        print(f"❌ チェックポイントが見つかりません: {ckpt_path}")
        return None

    _state = torch.load(base_model, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    print(f"✅ モデル読み込み完了")

    # --- データセット設定 ---
    meta_cfg['DATASET']['data_path'] = data_path
    OmegaConf.set_readonly(meta_cfg._dot_config, False)
    meta_cfg._dot_config.DATASET.data_path = data_path
    OmegaConf.set_readonly(meta_cfg._dot_config, True)

    test_dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    print(f"✅ データセット読み込み完了: {len(test_dataset)} サンプル")

    # --- 出力ディレクトリ ---
    output_dir = os.path.join("/root/GUAVA/ply_outputs", output_name)
    os.makedirs(output_dir, exist_ok=True)

    video_ids = list(test_dataset.videos_info.keys())
    results = []

    with torch.no_grad():
        for vidx, video_id in enumerate(video_ids):
            print(f"\n{'='*70}")
            print(f"処理中: {video_id} [{vidx+1}/{len(video_ids)}]")
            print(f"{'='*70}")

            video_output_dir = os.path.join(output_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)

            # ソース情報読み込み
            source_info = test_dataset._load_source_info(video_id)

            # Ubody Gaussians推論
            print("  🔄 Ubody Gaussians推論中...")
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
                print("  💾 点群PLYを保存中...")
                ubody_gaussians.save_point_ply(video_output_dir, save_split=save_split)
                ply_files.append('canonical.ply')
                if save_split:
                    ply_files.extend(['canonical_smplx.ply', 'canonical_uv.ply'])

            if save_gaussian:
                print("  💾 Gaussian PLY（公式）を保存中...")
                ubody_gaussians.save_gaussian_ply(video_output_dir, save_split=save_split)
                
                # 公式PLYファイルのパス
                official_ply = os.path.join(video_output_dir, 'GS_canonical.ply')
                
                # 三角形データを追加した新しいPLYファイル
                enhanced_ply = os.path.join(video_output_dir, 'GS_canonical_full.ply')
                
                # EHMメッシュから三角形データを取得
                print("  🔧 三角形データを追加中...")
                try:
                    faces = ubody_gaussians.smplx.faces_tensor.cpu().numpy()
                    num_faces = faces.shape[0]
                    print(f"    📐 EHMメッシュ: {num_faces:,} 三角形")
                    
                    # 三角形データを追加
                    add_faces_to_ply(official_ply, enhanced_ply, faces)
                    
                    print(f"    ✅ 三角形データ追加完了")
                    ply_files.append('GS_canonical_full.ply')
                    
                    # 検証
                    if verify_format:
                        print(f"\n  🔍 PLYファイル検証: GS_canonical_full.ply")
                        verification = verify_ply_format(enhanced_ply)
                        
                        if "error" in verification:
                            print(f"    ❌ エラー: {verification['error']}")
                        else:
                            print(f"    ✅ 頂点数: {verification['vertex_count']:,}")
                            print(f"    ✅ 三角形数: {verification['face_count']:,}")
                            print(f"    ✅ プロパティ数: {verification['property_count']}")
                            print(f"    📊 ファイルサイズ: {verification['file_size_mb']:.2f} MB")
                            print(f"    📐 データサイズ一致: {'✅' if verification['size_match'] else '❌'}")
                            print(f"    🔹 三角形データ: {'✅ あり' if verification['has_faces'] else '❌ なし'}")
                            print(f"    🔹 Opacity属性: {'✅ あり' if verification['has_opacity'] else '❌ なし'}")
                            print(f"    🔹 Rotation属性: {'✅ あり' if verification['has_rotation'] else '❌ なし'}")
                            print(f"    🔹 Scale属性: {'✅ あり' if verification['has_scale'] else '❌ なし'}")
                        
                        # サンプルデータ抽出
                        if extract_samples:
                            print(f"\n  📊 サンプルデータ抽出:")
                            extract_ply_sample(enhanced_ply, num_samples=3)
                    
                except Exception as e:
                    print(f"    ❌ 三角形データ追加エラー: {e}")
                    import traceback
                    traceback.print_exc()

            # gvrm.ts/ply.ts互換のWeb PLYを生成
            if save_web:
                print("  💾 Web PLY (gvrm.ts互換)を保存中...")
                try:
                    faces = ubody_gaussians.smplx.faces_tensor.cpu().numpy()
                    web_ply_path, web_num_points = save_web_compatible_ply(
                        ubody_gaussians, video_output_dir, faces=faces
                    )
                    ply_files.append('avatar_web.ply')
                    print(f"    ✅ avatar_web.ply 保存完了 ({web_num_points:,} 頂点)")

                    # Web PLYの検証
                    if verify_format:
                        print(f"\n  🔍 PLYファイル検証: avatar_web.ply")
                        web_verification = verify_ply_format(web_ply_path)
                        if "error" not in web_verification:
                            print(f"    ✅ 頂点数: {web_verification['vertex_count']:,}")
                            print(f"    ✅ 三角形数: {web_verification['face_count']:,}")
                            print(f"    ✅ プロパティ数: {web_verification['property_count']} (期待: 12)")
                            print(f"    📊 ファイルサイズ: {web_verification['file_size_mb']:.2f} MB")
                except Exception as e:
                    print(f"    ❌ Web PLY保存エラー: {e}")
                    import traceback
                    traceback.print_exc()

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
            
            # 検証結果を追加
            if verify_format and save_gaussian and os.path.exists(enhanced_ply):
                verification = verify_ply_format(enhanced_ply)
                if "error" not in verification:
                    result['verification'] = verification
            
            results.append(result)

            print(f"\n  ✅ 完了: Template={num_template:,}, UV={num_uv:,}, Total={total:,} ({infer_time*1000:.1f}ms)")

    # クリーンアップ
    test_dataset._lmdb_engine.close()

    # サマリー保存
    summary = {
        'data_path': data_path,
        'output_dir': output_dir,
        'total_videos': len(results),
        'format': 'official_gaussian_ply_with_faces',
        'paper_compliant': True,
        'has_triangles': True,
        'settings': {
            'save_split': save_split,
            'save_point_cloud': save_point_cloud,
            'save_gaussian': save_gaussian,
            'save_web': save_web,
            'verify_format': verify_format
        },
        'videos': results
    }

    summary_path = os.path.join(output_dir, 'generation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 生成完了!")
    print("=" * 70)
    print(f"📁 出力ディレクトリ: {output_dir}")
    print(f"📊 処理動画数: {len(results)}")
    print(f"📝 サマリー: {summary_path}")
    
    # 検証サマリー
    if verify_format and results:
        print("\n📋 検証サマリー:")
        for result in results:
            if 'verification' in result:
                v = result['verification']
                print(f"  {result['video_id']}:")
                print(f"    - 頂点: {v['vertex_count']:,}, 三角形: {v['face_count']:,}")
                print(f"    - 三角形データ: {'✅' if v['has_faces'] else '❌'}")
                print(f"    - 必須属性: Opacity={'✅' if v['has_opacity'] else '❌'}, "
                      f"Rotation={'✅' if v['has_rotation'] else '❌'}, "
                      f"Scale={'✅' if v['has_scale'] else '❌'}")

    # Volume コミット
    ply_output_volume.commit()
    print("\n✅ PLY files committed to Volume")

    return summary


@app.local_entrypoint()
def main(
    output_name: str = "guava_avatar",
    split: bool = False,
    point_cloud: bool = False,
    gaussian: bool = True,
    web: bool = True,
    web_only: bool = False,
    no_verify: bool = False,
    no_samples: bool = False
):
    """
    PLY生成のエントリーポイント

    使用例:
        modal run generate_ply_cloud.py
        modal run generate_ply_cloud.py --output-name my_avatar
        modal run generate_ply_cloud.py --split
        modal run generate_ply_cloud.py --web-only  # avatar_web.plyのみ生成
        modal run generate_ply_cloud.py --no-verify
    """
    print("=" * 70)
    print("🚀 GUAVA PLY Generator (三角形データ追加版)")
    print("📖 公式PLY + EHMメッシュの三角形データ + gvrm.ts互換Web PLY")
    print("=" * 70)

    # web_onlyの場合は他のPLYを無効化
    if web_only:
        gaussian = False
        point_cloud = False
        split = False
        web = True

    result = generate_ply.remote(
        output_name=output_name,
        save_split=split,
        save_point_cloud=point_cloud,
        save_gaussian=gaussian,
        save_web=web,
        verify_format=not no_verify,
        extract_samples=not no_samples
    )

    if result:
        print("\n" + "=" * 70)
        print("📊 生成結果サマリー")
        print("=" * 70)
        print(f"📁 出力ディレクトリ: {result['output_dir']}")
        print(f"📊 処理動画数: {result['total_videos']}")
        print(f"📖 フォーマット: {result['format']}")
        print(f"✅ 論文準拠: {result['paper_compliant']}")
        print(f"✅ 三角形データ: {result['has_triangles']}")
        for video in result['videos']:
            print(f"\n  📹 {video['video_id']}:")
            print(f"    🔹 Template Gaussians: {video['num_template_gaussians']:,}")
            print(f"    🔹 UV Gaussians: {video['num_uv_gaussians']:,}")
            print(f"    🔹 Total: {video['total_gaussians']:,}")
            print(f"    ⚡ 推論時間: {video['inference_time_ms']:.1f}ms")
            print(f"    📄 PLYファイル:")
            for ply in video['ply_files']:
                print(f"      - {ply}")
            
            if 'verification' in video:
                v = video['verification']
                print(f"    ✅ 検証:")
                print(f"      - 頂点数: {v['vertex_count']:,}")
                print(f"      - 三角形数: {v['face_count']:,}")
                print(f"      - ファイルサイズ: {v['file_size_mb']:.2f} MB")

        print("\n" + "=" * 70)
        print("✅ PLY生成が完了しました")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ PLY生成に失敗しました")
        print("=" * 70)