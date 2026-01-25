#!/usr/bin/env python3
"""
extract_refiner_weights.py
Extract Neural Refiner weights from PyTorch checkpoint to JSON for WebGL

Usage:
  python extract_refiner_weights.py best_160000.pt refiner_weights.json
"""

import torch
import json
import base64
import sys
import os

def to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64-encoded float32"""
    arr = tensor.detach().cpu().float().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_refiner_weights.py <checkpoint.pt> <output.json>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict") or
            checkpoint.get("model_state_dict") or
            checkpoint.get("model") or
            checkpoint
        )
    else:
        state_dict = checkpoint
    
    # Clean keys
    clean = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("refiner.", "")
        clean[k] = v
    
    print(f"Found {len(clean)} tensors")
    
    # Convert to JSON
    result = {}
    for key, tensor in clean.items():
        shape = list(tensor.shape)
        print(f"  {key}: {shape}")
        result[key] = to_base64(tensor)
    
    # Save
    print(f"\nSaving: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(result, f)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Size: {size_mb:.2f} MB")
    print("✅ Done!")

if __name__ == '__main__':
    main()
