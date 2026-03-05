#!/usr/bin/env python3
"""Upload Spoke training data to Modal Volume.

Usage:
    python spoke/cloud/upload_data.py
    python spoke/cloud/upload_data.py --data-dir spoke/data/v5
    python spoke/cloud/upload_data.py --no-include-bench
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

VOLUME_NAME = "spoke-training-data"
REMOTE_DIR = "/"


def modal_prefix() -> list[str]:
    """Prefer modal CLI; fall back to uvx modal."""
    if shutil.which("modal"):
        return ["modal"]
    if shutil.which("uvx"):
        return ["uvx", "modal"]
    print("Error: neither 'modal' nor 'uvx' is installed in PATH.")
    sys.exit(1)


def run_modal(prefix: list[str], args: list[str], capture: bool = True):
    return subprocess.run(
        prefix + args,
        capture_output=capture,
        text=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Upload training data to Modal volume")
    parser.add_argument("--data-dir", default="spoke/data/v5", help="Local data directory")
    parser.add_argument(
        "--include-bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also upload benchmark sets to /bench/ on the same volume",
    )
    args = parser.parse_args()

    prefix = modal_prefix()
    data_dir = Path(args.data_dir)
    files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    bench_dir = Path("spoke/bench")
    bench_files = [
        "test_set.json",
        "test_set_v3.json",
        "test_set_evals.json",
    ]

    # Verify files exist
    for f in files:
        path = data_dir / f
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        lines = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {f}: {lines} examples")

    if args.include_bench:
        print("\nBenchmark sets:")
        for f in bench_files:
            path = bench_dir / f
            if not path.exists():
                print(f"Error: {path} not found")
                sys.exit(1)
            with open(path, encoding="utf-8") as fh:
                count = len(json.load(fh))
            print(f"  {f}: {count} examples")

    # Create volume if it doesn't exist (idempotent)
    print(f"\nCreating volume '{VOLUME_NAME}' (if not exists)...")
    run_modal(prefix, ["volume", "create", VOLUME_NAME], capture=True)

    # Upload each file
    for f in files:
        local_path = str(data_dir / f)
        remote_path = f"/{f}"
        print(f"Uploading {local_path} -> {VOLUME_NAME}:{remote_path}")
        result = run_modal(
            prefix,
            ["volume", "put", "--force", VOLUME_NAME, local_path, remote_path],
            capture=True,
        )
        if result.returncode != 0:
            print(f"Error uploading {f}: {result.stderr}")
            sys.exit(1)

    if args.include_bench:
        for f in bench_files:
            local_path = str(bench_dir / f)
            remote_path = f"/bench/{f}"
            print(f"Uploading {local_path} -> {VOLUME_NAME}:{remote_path}")
            result = run_modal(
                prefix,
                ["volume", "put", "--force", VOLUME_NAME, local_path, remote_path],
                capture=True,
            )
            if result.returncode != 0:
                print(f"Error uploading {f}: {result.stderr}")
                sys.exit(1)

    print(f"\nDone! Files uploaded to volume '{VOLUME_NAME}'.")
    verify_cmd = " ".join(prefix + ["volume", "ls", VOLUME_NAME, "/"])
    print(f"Verify with: {verify_cmd}")


if __name__ == "__main__":
    main()
