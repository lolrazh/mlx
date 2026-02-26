"""Dataset loaders for ASR evaluation.

Yields normalized (audio_array, sample_rate, reference_text) tuples
from standard speech benchmarks.
"""

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()  # loads HF_TOKEN from .env

LIBRISPEECH_CLEAN = "librispeech-clean"
LIBRISPEECH_OTHER = "librispeech-other"

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
}


def load_samples(dataset_name, max_samples=None, shuffle=False, seed=42):
    """Yield (audio_array, sample_rate, reference_text) from a dataset.

    Args:
        dataset_name: One of the keys in DATASET_REGISTRY.
        max_samples: If set, only yield this many samples (useful for quick tests).
        shuffle: If True, shuffle the dataset before sampling. Important when
            using max_samples to avoid speaker/chapter bias.
        seed: Random seed for reproducible shuffling.

    Yields:
        (numpy array of float32 audio, int sample_rate, str reference_text)
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {available}")

    cfg = DATASET_REGISTRY[dataset_name]

    ds = load_dataset(
        cfg["path"],
        cfg["name"],
        split=cfg["split"],
    )

    if shuffle:
        ds = ds.shuffle(seed=seed)

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
