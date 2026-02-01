#!/usr/bin/env python3
"""
Modal Volumeからテスト結果の動画をダウンロードするスクリプト

使用方法:
    python download_video.py
"""

import modal
import os
import base64

# Volume設定
output_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10")

app = modal.App("download-video")

@app.function(
    image=image,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
    },
    timeout=600,
)
def get_video():
    """動画ファイルを取得"""
    results_path = "/root/EHM-Tracker/output/test_results"

    if not os.path.exists(results_path):
        return {"error": f"結果が見つかりません: {results_path}"}

    # 動画ファイルを検索
    videos = []
    for root, dirs, filenames in os.walk(results_path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                videos.append(os.path.join(root, filename))

    if not videos:
        return {"error": "動画ファイルが見つかりません"}

    video_path = videos[0]
    print(f"動画ファイル: {video_path}")

    with open(video_path, 'rb') as f:
        video_data = f.read()

    return {
        "success": True,
        "path": video_path,
        "filename": os.path.basename(video_path),
        "size": len(video_data),
        "data": base64.b64encode(video_data).decode('utf-8')
    }


@app.function(
    image=image,
    volumes={
        "/root/EHM-Tracker/output": output_volume,
    },
    timeout=300,
)
def list_results():
    """結果ファイル一覧を取得"""
    results_path = "/root/EHM-Tracker/output/test_results"

    if not os.path.exists(results_path):
        return {"error": f"結果が見つかりません: {results_path}"}

    files = []
    for root, dirs, filenames in os.walk(results_path):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, results_path)
            files.append({
                "path": rel_path,
                "size": os.path.getsize(full_path)
            })
    return {"files": files}


@app.local_entrypoint()
def main(list_only: bool = False):
    if list_only:
        print("=== 結果ファイル一覧 ===")
        result = list_results.remote()
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        for f in result['files']:
            print(f"  {f['path']}: {f['size'] / 1024:.1f} KB")
        return

    print("=== 動画ダウンロード ===")
    result = get_video.remote()

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # ローカルに保存
    os.makedirs('test_results', exist_ok=True)
    local_path = os.path.join('test_results', result['filename'])
    with open(local_path, 'wb') as f:
        f.write(base64.b64decode(result['data']))

    print(f"動画保存完了: {local_path}")
    print(f"サイズ: {result['size'] / 1024:.1f} KB")
    print(f"元のパス: {result['path']}")


if __name__ == "__main__":
    # modal run download_video.py で実行
    pass
