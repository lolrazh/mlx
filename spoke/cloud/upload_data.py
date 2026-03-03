#!/usr/bin/env python3
"""Upload Spoke training data to Modal Volume.

Usage:
    python spoke/cloud/upload_data.py
    python spoke/cloud/upload_data.py --data-dir spoke/data/v4
"""

import argparse
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "spoke-training-data"
REMOTE_DIR = "/"


def main():
    parser = argparse.ArgumentParser(description="Upload training data to Modal volume")
    parser.add_argument("--data-dir", default="spoke/data/v4", help="Local data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = ["train.jsonl", "valid.jsonl", "test.jsonl"]

    # Verify files exist
    for f in files:
        path = data_dir / f
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        lines = sum(1 for _ in open(path))
        print(f"  {f}: {lines} examples")

    # Create volume if it doesn't exist (idempotent)
    print(f"\nCreating volume '{VOLUME_NAME}' (if not exists)...")
    subprocess.run(["modal", "volume", "create", VOLUME_NAME], capture_output=True)

    # Upload each file
    for f in files:
        local_path = str(data_dir / f)
        remote_path = f"/{f}"
        print(f"Uploading {local_path} -> {VOLUME_NAME}:{remote_path}")
        result = subprocess.run(
            ["modal", "volume", "put", VOLUME_NAME, local_path, remote_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error uploading {f}: {result.stderr}")
            sys.exit(1)

    print(f"\nDone! Files uploaded to volume '{VOLUME_NAME}'.")
    print(f"Verify with: modal volume ls {VOLUME_NAME} /")


if __name__ == "__main__":
    main()
