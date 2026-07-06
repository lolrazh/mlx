# Emoji ablation (2026-07-06)

Tests the hypothesis (user, 2026-07-06): does removing emoji examples from
training give more reliable output on the *other* categories, because the
adapter's limited LoRA capacity no longer spends ~10% of itself on a
segregated sub-task (per finding #102: emoji fires an isolated expert circuit)?

Variants (from spoke/data/v5):
- train_full.jsonl    1046 (= v5 train, unchanged)
- train_noemoji.jsonl  940 (106 emoji-output rows removed)
- valid_full.jsonl     131
- valid_noemoji.jsonl  119 (12 emoji rows removed)

Design: both runs use valid_noemoji as the checkpoint-selection yardstick
(identical), so the ONLY variable is the 106 emoji rows in training.
Recipe = champion (Qwen3-4B-Instruct-2507, r8/a16, lr1e-5, adam, constant,
batch4, seq256, 2000 steps). Compare on NON-emoji broad58 categories.
Confound: noemoji trains on 106 fewer examples (a handicap) — so if the
smaller set still WINS on non-emoji categories, that's strong confirmation.

Runs: spoke-abl-full, spoke-abl-noemoji (Modal sandy-36852).
