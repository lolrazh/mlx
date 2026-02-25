# MLX Learning Roadmap

> **Goal:** Take any model and port it to MLX, squeezing max performance on Apple Silicon.
>
> **Your setup:** M4, 24 GB unified memory · macOS · Python venv with `mlx`, `mlx-lm`, `transformers`, `safetensors`
>
> **Starting point:** You've watched 3b1b's neural network series + transformer video. You know what weights, matmul, gradient descent, backprop, and attention are *conceptually*. You've never coded a model or touched model files.

---

## How to use this roadmap

1. Work through lessons in order — each builds on the last
2. Run every `exercises.py` in your terminal (`python exercises.py`)
3. After each milestone, experiment in `playground/` before moving on
4. The visualizer (`visualizer/index.html`) can be opened anytime for interactive reference
5. Check off each lesson as you complete it

---

## Milestone 1 — Foundations

*The toolbox. Learn the primitives before you build anything real.*

- [ ] **[Lesson 01 — MLX Core](lessons/01-mlx-core/README.md)**
  Arrays, dtypes, lazy evaluation, `mx.grad()`. The building blocks of everything.
  > **Analogy:** Arrays are spreadsheet cells that live on the GPU. MLX is lazy — it writes down the formula but doesn't calculate until you press Enter.

- [ ] **[Lesson 02 — nn.Module](lessons/02-nn-modules/README.md)**
  Neural network building blocks: Linear layers, models as classes, weight trees.
  > **Analogy:** Modules are LEGO bricks. A `Linear` layer is one brick. A model is a structure built by snapping bricks together. Each brick stores its weights inside.

**After this milestone you can:** Create arrays, do matrix math on the GPU, build a simple neural network class, and inspect its weights.

---

## Milestone 2 — Real Models

*Open the hood. See what an actual LLM looks like on disk.*

- [ ] **[Lesson 03 — Inside a Model](lessons/03-inside-a-model/README.md)**
  Download a real model, crack open safetensors + config.json, inspect every weight.
  > **Analogy:** A HuggingFace model is a zip file containing a blueprint (`config.json`), the bricks themselves (`safetensors`), and a dictionary (`tokenizer`).

- [ ] **[Lesson 04 — The Transformer](lessons/04-the-transformer/README.md)**
  Build a transformer block from scratch in MLX. Attention, MLP, layer norms.
  > **Analogy:** Attention is a room full of people (tokens). Each person looks around, decides who's relevant, and updates their understanding. The Q/K/V matrices are how they "look" and "listen."

**After this milestone you can:** Download any model from HuggingFace, list every weight name and shape, and understand what each weight does in the transformer architecture.

---

## Milestone 3 — Porting (The Core Goal)

*The skill you came here for. Take a model from anywhere and make it run in MLX.*

- [ ] **[Lesson 05 — Conversion Pipeline](lessons/05-conversion-pipeline/README.md)**
  PyTorch → MLX weight mapping. Manual conversion, then `mlx_lm.convert`.
  > **Analogy:** Converting a model is translating a book — same story, different language. The "translation" is mostly renaming chapter titles (weight keys).

- [ ] **[Lesson 06 — Quantization](lessons/06-quantization/README.md)**
  Float16 → 4-bit. How it works, quality tradeoffs, speed gains.
  > **Analogy:** Quantization is switching from a RAW photo to a compressed JPEG. Tiny quality loss, 4x smaller, 4x faster to load. For LLMs, "faster to load" = more tokens per second.

**After this milestone you can:** Take a PyTorch model from HuggingFace, convert it to MLX format, quantize it to 4-bit, and run it locally. This is the whole game.

---

## Milestone 4 — Performance

*Squeeze every last token/second from your M4.*

- [ ] **[Lesson 07 — Performance](lessons/07-performance/README.md)**
  Memory bandwidth ceilings, KV cache, speculative decoding, prompt caching. Squeezing every tok/s from your M4.
  > **Analogy:** Your M4 reads weights from memory like a librarian fetching books from shelves. Memory bandwidth (120 GB/s) is how fast the librarian walks. Quantization makes the books smaller. Speculative decoding lets a fast intern pre-fetch books the librarian will probably need.

**After this milestone you can:** Benchmark different configurations, understand your hardware ceiling, and know exactly which knobs to turn for maximum speed.

---

## Milestone 5 — Visualizer

- [ ] **[Interactive Visualizer](visualizer/index.html)**
  Single-page web app with tabs:
  - Matrix multiply animation (how weights transform data)
  - Attention heatmap (how tokens attend to each other)
  - Quantization slider (see precision loss in real-time)
  - Memory layout (Apple unified memory vs NVIDIA discrete)

---

## Quick Reference

| Concept | Where you learned it (3b1b) | Where you'll code it |
|---|---|---|
| Weights & biases | Neural Networks #1 | Lessons 01, 02 |
| Matrix multiplication | Neural Networks #1 | Lesson 01 |
| Gradient descent | Neural Networks #2 | Lesson 01 |
| Backpropagation | Neural Networks #3 | Lesson 01 (`mx.grad`) |
| Attention (Q, K, V) | Transformers | Lesson 04 |
| Softmax | Transformers | Lesson 04 |

---

## Folder Layout

```
mlx/
├── ROADMAP.md              ← you are here
├── lessons/
│   ├── 01-mlx-core/        ← arrays, matmul, grad
│   ├── 02-nn-modules/      ← building layers, weight trees
│   ├── 03-inside-a-model/  ← crack open a real LLM
│   ├── 04-the-transformer/ ← build attention from scratch
│   ├── 05-conversion-pipeline/ ← HuggingFace → MLX
│   ├── 06-quantization/    ← float16 → 4-bit
│   └── 07-performance/     ← benchmarks, optimization
├── visualizer/
│   └── index.html          ← interactive visual reference
└── playground/             ← your scratch space
```

---

*Built for learning. Break things. Read the errors. That's the point.*
