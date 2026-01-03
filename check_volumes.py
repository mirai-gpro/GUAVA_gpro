"""Check Modal Volumes for concierge assets"""
import modal
import os

ehm_volume = modal.Volume.from_name("ehm-tracker-output")
weights_volume = modal.Volume.from_name("guava-weights")
guava_volume = modal.Volume.from_name("guava-results", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10")

app = modal.App("check-volumes")

@app.function(
    image=image,
    volumes={
        "/ehm": ehm_volume,
        "/weights": weights_volume,
        "/results": guava_volume
    },
    timeout=120
)
def check_volumes():
    import os

    def list_dir(path, depth=0, max_depth=3):
        if depth > max_depth:
            return
        try:
            items = os.listdir(path)
            for item in sorted(items)[:20]:  # Limit to 20 items
                full_path = os.path.join(path, item)
                prefix = "  " * depth
                if os.path.isdir(full_path):
                    print(f"{prefix}📁 {item}/")
                    list_dir(full_path, depth + 1, max_depth)
                else:
                    size = os.path.getsize(full_path)
                    print(f"{prefix}📄 {item} ({size:,} bytes)")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*60)
    print("📦 EHM-TRACKER-OUTPUT Volume (/ehm)")
    print("="*60)
    list_dir("/ehm")

    print("\n" + "="*60)
    print("📦 GUAVA-WEIGHTS Volume (/weights)")
    print("="*60)
    list_dir("/weights")

    print("\n" + "="*60)
    print("📦 GUAVA-RESULTS Volume (/results)")
    print("="*60)
    list_dir("/results")

@app.local_entrypoint()
def main():
    check_volumes.remote()
