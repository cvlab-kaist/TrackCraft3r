"""
Download Wan2.1-T2V-1.3B weights from Hugging Face into the same layout that
the training scripts expect under MODELSCOPE_CACHE.

Files fetched:
  - diffusion_pytorch_model*.safetensors  (DiT)
  - models_t5_umt5-xxl-enc-bf16.pth       (T5 text encoder)
  - Wan2.1_VAE.pth                        (VAE)

Usage:
    pip install -U huggingface_hub
    python scripts/download_wan_1.3B.py [--target ./checkpoints/wan_models]
or set MODELSCOPE_CACHE first:
    MODELSCOPE_CACHE=./checkpoints/wan_models python scripts/download_wan_1.3B.py
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
ALLOW_PATTERNS = [
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "config.json",
    "configuration.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default=os.environ.get("MODELSCOPE_CACHE", "./checkpoints/wan_models"),
        help="Root directory for cached model weights (matches MODELSCOPE_CACHE).",
    )
    args = parser.parse_args()

    local_dir = Path(args.target) / REPO_ID
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} into {local_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(local_dir),
        allow_patterns=ALLOW_PATTERNS,
    )
    print("Done.")


if __name__ == "__main__":
    main()
