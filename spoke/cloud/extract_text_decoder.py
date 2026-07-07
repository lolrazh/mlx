#!/usr/bin/env python3
"""Extract the text decoder from a merged Gemma 4 multimodal checkpoint.

The Gemma 4 merge (merge_adapter_checkpoint.py) saves the FULL
`Gemma4ForConditionalGeneration` — text decoder + vision tower + audio tower
(~16 GB bf16). For ASR post-processing we only ever run the text decoder, and
MLX's `gemma4.sanitize()` drops the vision/audio weights at load time anyway.

This function strips those towers ON THE VOLUME so the local download is only
the ~9 GB language model, not the full multimodal artifact. The key filter
mirrors mlx_lm/models/gemma4.py::sanitize exactly: keep `model.language_model.*`
and any top-level (non-`model.`) tensor such as `lm_head`; drop everything under
`model.vision_tower / audio_tower / multi_modal_projector / embed_*`.

This is CLAUDE.md Phase 2's "weight sanitize function", done cloud-side.

Usage:
    # Inspect only — report key partitions + byte split, write nothing:
    SPOKE_GEMMA4_IMAGE=1 modal run spoke/cloud/extract_text_decoder.py \
      --run-name spoke-g4e4b-hp-lplus2-20260707-ckpt900 --inspect-only

    # Extract the text decoder to <run>/text-decoder on the volume:
    SPOKE_GEMMA4_IMAGE=1 modal run spoke/cloud/extract_text_decoder.py \
      --run-name spoke-g4e4b-hp-lplus2-20260707-ckpt900
"""

from __future__ import annotations

import modal

app = modal.App("spoke-extract-text-decoder")

output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

# Light image: only safetensors + torch (needed to round-trip bf16 tensors,
# which numpy cannot represent). No GPU.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.*", "safetensors")
)

# Keys mlx_lm/models/gemma4.py::sanitize discards. Kept in sync with that file.
_DROP_UNDER_MODEL = (
    "vision_tower",
    "audio_tower",
    "multi_modal_projector",
    "embed_audio",
    "embed_vision",
)


def _classify(key: str) -> str:
    """Return 'keep' or a drop-reason for a raw HF checkpoint key."""
    if key.startswith("model."):
        sub = key[len("model.") :]
        if sub.startswith(_DROP_UNDER_MODEL):
            return sub.split(".")[0]  # tower name
        if sub.startswith("language_model"):
            return "keep"
        # Any other model.* (rare) — keep and let mlx decide.
        return "keep"
    # Top-level tensors (e.g. lm_head) — mlx keeps these as-is.
    return "keep"


@app.function(
    image=image,
    volumes={"/output": output_vol},
    memory=32768,
    timeout=1800,
)
def extract(run_name: str, inspect_only: bool = False):
    import json
    import shutil
    from collections import defaultdict
    from pathlib import Path

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    merged_dir = Path(f"/output/{run_name}/merged")
    st_path = merged_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"No merged safetensors at {st_path}")

    # --- Pass 1: read the header only, partition keys + byte sizes ---------
    _DTYPE_BYTES = {
        "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }
    import struct

    with open(st_path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))

    part_bytes: dict[str, int] = defaultdict(int)
    part_count: dict[str, int] = defaultdict(int)
    keep_keys: list[str] = []
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        reason = _classify(k)
        n = 1
        for d in meta["shape"]:
            n *= d
        nbytes = n * _DTYPE_BYTES.get(meta["dtype"], 2)
        bucket = "language_model (KEEP)" if reason == "keep" else f"{reason} (drop)"
        part_bytes[bucket] += nbytes
        part_count[bucket] += 1
        if reason == "keep":
            keep_keys.append(k)

    print(f"\n=== {run_name} / merged / model.safetensors ===")
    total = sum(part_bytes.values())
    for bucket in sorted(part_bytes, key=lambda b: -part_bytes[b]):
        gb = part_bytes[bucket] / 1e9
        print(f"  {bucket:32s} {part_count[bucket]:5d} tensors  {gb:6.2f} GB")
    print(f"  {'TOTAL':32s} {sum(part_count.values()):5d} tensors  {total/1e9:6.2f} GB")
    keep_gb = part_bytes["language_model (KEEP)"] / 1e9
    print(f"\n  Download shrinks {total/1e9:.1f} GB -> {keep_gb:.1f} GB "
          f"({100*keep_gb/total:.0f}% of full).")

    # Report tie_word_embeddings / lm_head presence — affects mlx head handling.
    cfg = json.loads((merged_dir / "config.json").read_text())
    has_lm_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight")
                      for k in header if k != "__metadata__")
    print(f"  tie_word_embeddings={cfg.get('tie_word_embeddings')} | "
          f"lm_head weight present={has_lm_head}")

    if inspect_only:
        print("\n[inspect-only] wrote nothing.")
        return

    # --- Pass 2: stream kept tensors into a new text-decoder checkpoint -----
    out_dir = Path(f"/output/{run_name}/text-decoder")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(keep_keys)} tensors -> {out_dir}")

    kept: dict[str, torch.Tensor] = {}
    with safe_open(str(st_path), framework="pt") as f:
        for k in keep_keys:
            kept[k] = f.get_tensor(k)
    save_file(kept, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    # Copy the sidecar files mlx_lm.convert needs. Keep the FULL gemma4
    # config.json — mlx reads text_config and ignores vision/audio blocks.
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"):
        src = merged_dir / name
        if src.exists():
            shutil.copy(str(src), str(out_dir / name))

    output_vol.commit()
    out_gb = (out_dir / "model.safetensors").stat().st_size / 1e9
    print(f"Done. text-decoder = {out_gb:.2f} GB at {out_dir}")


@app.local_entrypoint()
def main(run_name: str, inspect_only: bool = False):
    extract.remote(run_name=run_name, inspect_only=inspect_only)
