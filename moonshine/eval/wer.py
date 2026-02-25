"""Word Error Rate computation with text normalization.

Wraps jiwer with consistent preprocessing so all models
are compared on the same footing.
"""

import re
import jiwer


def normalize_text(text):
    """Lowercase, strip punctuation, collapse whitespace.

    This is the standard ASR normalization — we compare words only,
    ignoring casing and punctuation differences.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def compute_wer(references, hypotheses):
    """Compute WER and error breakdown over a list of (reference, hypothesis) pairs.

    Args:
        references: List of reference transcription strings.
        hypotheses: List of hypothesis transcription strings.

    Returns:
        dict with keys:
            wer: float, overall word error rate
            substitutions: int, total substitutions
            insertions: int, total insertions
            deletions: int, total deletions
            total_words: int, total reference words
            num_samples: int, number of samples
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )

    # Normalize all text
    norm_refs = [normalize_text(r) for r in references]
    norm_hyps = [normalize_text(h) for h in hypotheses]

    # Skip empty references (jiwer can't compute WER for empty strings)
    filtered = [
        (r, h) for r, h in zip(norm_refs, norm_hyps) if len(r.split()) > 0
    ]
    if not filtered:
        return {
            "wer": 0.0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "total_words": 0,
            "num_samples": 0,
        }

    norm_refs, norm_hyps = zip(*filtered)
    norm_refs, norm_hyps = list(norm_refs), list(norm_hyps)

    output = jiwer.process_words(norm_refs, norm_hyps)

    return {
        "wer": output.wer,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "total_words": sum(len(r.split()) for r in norm_refs),
        "num_samples": len(norm_refs),
    }


def compute_wer_per_sample(references, hypotheses):
    """Compute WER for each individual sample.

    Returns:
        List of dicts, each with 'wer', 'reference', 'hypothesis'.
    """
    results = []
    for ref, hyp in zip(references, hypotheses):
        norm_ref = normalize_text(ref)
        norm_hyp = normalize_text(hyp)

        if len(norm_ref.split()) == 0:
            results.append({"wer": 0.0, "reference": ref, "hypothesis": hyp})
            continue

        sample_wer = jiwer.wer(norm_ref, norm_hyp)
        results.append({"wer": sample_wer, "reference": ref, "hypothesis": hyp})

    return results


def print_summary(results, dataset_name=""):
    """Print a formatted WER summary."""
    header = f"WER Results: {dataset_name}" if dataset_name else "WER Results"
    print(f"\n{'=' * 50}")
    print(f"  {header}")
    print(f"{'=' * 50}")
    print(f"  Samples evaluated:  {results['num_samples']}")
    print(f"  Total ref words:    {results['total_words']}")
    print(f"  WER:                {results['wer']:.2%}")
    print(f"  Substitutions:      {results['substitutions']}")
    print(f"  Insertions:         {results['insertions']}")
    print(f"  Deletions:          {results['deletions']}")
    print(f"{'=' * 50}\n")
