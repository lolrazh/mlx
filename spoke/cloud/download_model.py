#!/usr/bin/env python3
"""Download merged model from Modal Volume.

Usage:
    python spoke/cloud/download_model.py --run-name spoke-qwen35-t1
    python spoke/cloud/download_model.py --run-name spoke-qwen35-t1 --output spoke/models/qwen35-t1-bf16
"""

import argparse
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "spoke-output"


def main():
    parser = argparse.ArgumentParser(description="Download merged model from Modal")
    parser.add_argument("--run-name", required=True, help="Training run name (used as subfolder)")
    parser.add_argument("--output", default=None, help="Local output directory (default: spoke/models/<run-name>-bf16)")
    args = parser.parse_args()

    output_dir = args.output or f"spoke/models/{args.run_name}-bf16"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    remote_path = f"{args.run_name}/merged"

    print(f"Listing remote files at {VOLUME_NAME}:{remote_path}...")
    result = subprocess.run(
        ["modal", "volume", "ls", VOLUME_NAME, remote_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: Could not list {remote_path}")
        print(f"  {result.stderr.strip()}")
        print(f"\nAvailable runs:")
        subprocess.run(["modal", "volume", "ls", VOLUME_NAME, "/output"])
        sys.exit(1)

    print(f"Remote files:\n{result.stdout}")

    print(f"Downloading {VOLUME_NAME}:{remote_path}/ -> {output_dir}/")
    result = subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, remote_path, output_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        sys.exit(1)

    print(f"\nModel downloaded to: {output_dir}")

    # Check if key files are present
    expected = ["config.json", "tokenizer.json"]
    for f in expected:
        if not (output_path / f).exists():
            # Files might be in a subfolder from modal volume get
            print(f"  Warning: {f} not found at top level — check subdirectories")

    mlx_path = output_dir.replace("-bf16", "-mlx")
    dwq_path = output_dir.replace("-bf16", "-dwq4")
    print(f"""
Next steps:
  1. Convert to MLX format:
     mlx_lm.convert --hf-path {output_dir} --mlx-path {mlx_path}

  2. (Optional) DWQ 4-bit quantization:
     mlx_lm.dwq --hf-path {mlx_path} --mlx-path {dwq_path} \\
       --data-path spoke/data/v4/train.jsonl --grad-checkpoint --batch-size 1

  3. Benchmark:
     python spoke/bench/run_benchmark.py --model {mlx_path} --prompt-mode v2 --test-set spoke/bench/test_set_v3.json
""")


if __name__ == "__main__":
    main()
