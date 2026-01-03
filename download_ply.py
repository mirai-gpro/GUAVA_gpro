"""Download GS_canonical.ply from Modal Volume"""
import modal
import os

guava_volume = modal.Volume.from_name("guava-results")

app = modal.App("download-ply")

@app.function(
    volumes={"/results": guava_volume},
    timeout=300
)
def download_ply():
    import shutil

    src = "/results/driving_avatar/render_self_act/driving/GS_canonical.ply"

    if os.path.exists(src):
        # Read file content
        with open(src, "rb") as f:
            data = f.read()
        print(f"Read {len(data):,} bytes from {src}")
        return data
    else:
        print(f"File not found: {src}")
        return None

@app.local_entrypoint()
def main():
    data = download_ply.remote()
    if data:
        output_path = "assets/GS_canonical.ply"
        os.makedirs("assets", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        print(f"Saved to {output_path} ({len(data):,} bytes)")
