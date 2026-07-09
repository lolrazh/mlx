"""Nemotron 3.5 ASR streaming prototype: live partial transcripts on Apple Silicon.

Feeds audio to the cache-aware FastConformer-RNNT incrementally (paced file
playback or live mic) and reports partial-transcript cadence, emission latency,
prefix stability, and time-to-final. This is the feasibility probe for streaming
dictation in Spoke's pill UI.

Usage (from the repo root, .venv/bin/python):
    python moonshine/streaming/nemotron_stream_proto.py --wav clip.wav
    python moonshine/streaming/nemotron_stream_proto.py --wav clip.wav --lookahead 80
    python moonshine/streaming/nemotron_stream_proto.py --mic --duration 15

Lookahead maps to att_context_size: 80ms=[56,0], 320ms=[56,3], 560ms=[56,6],
1120ms=[56,13] (the trained configurations of nemotron-3.5-asr-streaming-0.6b).
Partials are cumulative and append-only (greedy RNNT never retracts a token).
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram_frames
from mlx_audio.stt.models.nemotron_asr.streaming import stream_encode_chunks
from mlx_audio.stt.utils import load_audio, load_model

DEFAULT_MODEL = "moonshine/models/nemotron-3.5-asr-streaming-0.6b"

LOOKAHEAD_TO_ACS = {80: [56, 0], 320: [56, 3], 560: [56, 6], 1120: [56, 13]}


class PacedFileFeed:
    """Yields (chunk, arrival_time) from a file at real-time pace."""

    def __init__(self, path, sample_rate, chunk_ms, seconds=None):
        audio = load_audio(path, sample_rate, dtype=mx.float32)
        if seconds is not None:
            audio = audio[: int(seconds * sample_rate)]
        self.audio = np.array(audio)
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.duration = len(self.audio) / sample_rate

    def __iter__(self):
        start = time.perf_counter()
        pos = 0
        n = 0
        while pos < len(self.audio):
            n += 1
            target = start + n * self.chunk_samples / self.sample_rate
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            chunk = self.audio[pos : pos + self.chunk_samples]
            pos += self.chunk_samples
            yield chunk, time.perf_counter()


class MicFeed:
    """Yields (chunk, arrival_time) from the default input device."""

    def __init__(self, sample_rate, chunk_ms, duration):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.duration = duration

    def __iter__(self):
        import sounddevice as sd

        q = queue.Queue()

        def callback(indata, frames, t, status):
            if status:
                print(f"[mic] {status}", file=sys.stderr)
            q.put((indata[:, 0].copy(), time.perf_counter()))

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=callback,
        )
        deadline = time.perf_counter() + self.duration
        print(f"[mic] recording {self.duration:.0f}s... speak now")
        with stream:
            while time.perf_counter() < deadline:
                try:
                    yield q.get(timeout=0.5)
                except queue.Empty:
                    continue
        print("[mic] done")


class IncrementalMel:
    """Frame-exact incremental log-mel over a growing PCM buffer.

    Emits only frames whose full STFT window is available, so streamed frames
    are bit-identical to an offline log_mel_spectrogram pass; flush() emits the
    zero-padded tail frames to match iter_log_mel_spectrogram's total count.
    """

    def __init__(self, preprocess_args):
        self.args = preprocess_args
        self.buffer = np.zeros(0, dtype=np.float32)
        self.emitted_frames = 0

    def _ready_frames(self):
        usable = len(self.buffer) - self.args.n_fft // 2
        if usable < 0:
            return 0
        return usable // self.args.hop_length + 1

    def push(self, chunk):
        self.buffer = np.concatenate([self.buffer, chunk])
        ready = self._ready_frames()
        if ready <= self.emitted_frames:
            return None
        mel = log_mel_spectrogram_frames(
            mx.array(self.buffer), self.args, self.emitted_frames, ready
        )
        self.emitted_frames = ready
        return mel

    def flush(self):
        total = len(self.buffer) // self.args.hop_length + 1
        if total <= self.emitted_frames:
            return None
        mel = log_mel_spectrogram_frames(
            mx.array(self.buffer), self.args, self.emitted_frames, total
        )
        self.emitted_frames = total
        return mel


def run_stream(model, feed, language, att_context_size, stats):
    """Pipe a PCM feed through incremental mel -> cache-aware encoder -> RNNT.

    Latency ("lag") per partial = wall time of its emission minus the real-time
    position of the last audio it covers, i.e. how far behind live speech the
    displayed text runs. Includes lookahead wait, chunking, and compute.
    """
    mel = IncrementalMel(model.preprocessor_config)

    def mel_chunks():
        for chunk, _arrived in feed:
            m = mel.push(chunk)
            if m is not None:
                yield m
        stats["feed_end"] = time.perf_counter()
        tail = mel.flush()
        if tail is not None:
            yield tail

    frame_sec = (
        model.encoder_config.subsampling_factor
        * model.preprocessor_config.hop_length
        / model.preprocessor_config.sample_rate
    )

    covered = []  # per encoder chunk: seconds of audio covered once decoded

    def tracked(prompted):
        frames = 0
        for chunk in prompted:
            frames += chunk.shape[1]
            covered.append(frames * frame_sec)
            yield chunk

    prompted = tracked(
        stream_encode_chunks(
            model, mel_chunks(), language, att_context_size=att_context_size
        )
    )

    prev_text = ""
    stats["partials"] = []
    stats["prefix_violations"] = 0
    start = stats["start"]
    for i, result in enumerate(model._decode_prompted_chunks(prompted)):
        now = time.perf_counter()
        text = result.text
        if not text.startswith(prev_text):
            stats["prefix_violations"] += 1
        delta = text[len(prev_text) :] if text.startswith(prev_text) else text
        lag = (now - start) - covered[i]
        stats["partials"].append(
            {"wall": now, "lag": lag, "text": text, "delta": delta}
        )
        if delta or not text:
            print(f"  [+{now - start:6.2f}s  lag {lag * 1000:5.0f}ms]  {delta!r}")
        prev_text = text
    stats["final_text"] = prev_text
    stats["final_time"] = time.perf_counter()
    stats["frame_sec"] = frame_sec


def summarize(stats, label):
    partials = stats["partials"]
    lags = sorted(p["lag"] for p in partials)
    if not lags:
        print("no partials emitted")
        return
    p50 = lags[len(lags) // 2]
    p95 = lags[int(len(lags) * 0.95)]
    ttf = stats["final_time"] - stats.get("feed_end", stats["final_time"])
    print(f"\n=== {label} ===")
    print(f"partials emitted:       {len(partials)}")
    print(f"text lag behind live:   p50 {p50 * 1000:.0f}ms  p95 {p95 * 1000:.0f}ms  max {lags[-1] * 1000:.0f}ms")
    print(f"time-to-final:          {ttf * 1000:.0f}ms after feed end")
    print(f"prefix violations:      {stats['prefix_violations']} (0 = append-only, no flicker)")
    print(f"final text:             {stats['final_text']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="Audio file, fed at real-time pace")
    src.add_argument("--mic", action="store_true", help="Live microphone input")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="en-US", help="Prompt key or 'auto'")
    parser.add_argument(
        "--lookahead", type=int, default=320, choices=sorted(LOOKAHEAD_TO_ACS),
        help="Encoder lookahead in ms (chunk size = latency floor)",
    )
    parser.add_argument("--chunk-ms", type=int, default=40, help="PCM feed granularity")
    parser.add_argument("--seconds", type=float, help="Trim --wav to this length")
    parser.add_argument("--duration", type=float, default=15, help="Mic recording length")
    parser.add_argument("--json-out", help="Write stats JSON here")
    args = parser.parse_args()

    acs = LOOKAHEAD_TO_ACS[args.lookahead]
    print(f"Loading {args.model} (lookahead {args.lookahead}ms, acs={acs})...")
    model = load_model(args.model)
    sr = model.preprocessor_config.sample_rate

    # Warm-up: compile/dispatch paths once so the paced run measures steady state.
    warm = {"start": time.perf_counter()}
    silent = [(np.zeros(sr // 10, dtype=np.float32), time.perf_counter())] * 12
    run_stream(model, silent, args.language, acs, warm)
    print("Warm-up complete.\n")

    if args.mic:
        feed = MicFeed(sr, args.chunk_ms, args.duration)
        label = f"mic {args.duration:.0f}s, lookahead {args.lookahead}ms"
    else:
        feed = PacedFileFeed(args.wav, sr, args.chunk_ms, args.seconds)
        label = f"{Path(args.wav).name} ({feed.duration:.1f}s), lookahead {args.lookahead}ms"
        print(f"Streaming {feed.duration:.1f}s of audio at real-time pace...")

    stats = {"start": time.perf_counter()}
    run_stream(model, feed, args.language, acs, stats)
    summarize(stats, label)

    if args.json_out:
        out = {
            "label": label,
            "lookahead_ms": args.lookahead,
            "partials": [
                {k: v for k, v in p.items() if k != "wall"} for p in stats["partials"]
            ],
            "prefix_violations": stats["prefix_violations"],
            "time_to_final_ms": (stats["final_time"] - stats.get("feed_end", stats["final_time"])) * 1000,
            "final_text": stats["final_text"],
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"stats written to {args.json_out}")


if __name__ == "__main__":
    main()
