"""Dataset loaders for ASR evaluation.

Yields normalized (audio_array, sample_rate, reference_text) tuples
from standard speech benchmarks.
"""

from datasets import load_dataset

LIBRISPEECH_CLEAN = "librispeech-clean"
LIBRISPEECH_OTHER = "librispeech-other"
COMMON_VOICE_EN = "common-voice-en"

DATASET_REGISTRY = {
    LIBRISPEECH_CLEAN: {
        "path": "openslr/librispeech_asr",
        "name": "clean",
        "split": "test",
        "text_key": "text",
    },
    LIBRISPEECH_OTHER: {
        "path": "openslr/librispeech_asr",
        "name": "other",
        "split": "test",
        "text_key": "text",
    },
    COMMON_VOICE_EN: {
        "path": "mozilla-foundation/common_voice_17_0",
        "name": "en",
        "split": "test",
        "text_key": "sentence",
    },
}


def load_samples(dataset_name, max_samples=None):
    """Yield (audio_array, sample_rate, reference_text) from a dataset.

    Args:
        dataset_name: One of the keys in DATASET_REGISTRY.
        max_samples: If set, only yield this many samples (useful for quick tests).

    Yields:
        (numpy array of float32 audio, int sample_rate, str reference_text)
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {available}")

    cfg = DATASET_REGISTRY[dataset_name]

    # Common Voice needs HF token for auth
    use_token = dataset_name == COMMON_VOICE_EN

    ds = load_dataset(
        cfg["path"],
        cfg["name"],
        split=cfg["split"],
        trust_remote_code=True,
        token=use_token or None,
    )

    for i, sample in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        audio = sample["audio"]
        text = sample[cfg["text_key"]]

        yield audio["array"], audio["sampling_rate"], text


def dataset_info(dataset_name):
    """Return metadata about a dataset (for logging)."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    cfg = DATASET_REGISTRY[dataset_name]
    return {
        "name": dataset_name,
        "hf_path": f"{cfg['path']}/{cfg['name']}",
        "split": cfg["split"],
    }


def list_datasets():
    """Return list of available dataset names."""
    return list(DATASET_REGISTRY.keys())
