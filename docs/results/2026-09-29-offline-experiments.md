# Offline experiments (2026-09-29)

Run on a 2-core CPU with no GPU and no access to Hugging Face, so no real teacher or text data was available. These runs test the architecture's mechanics, not embedding quality. Real STS results need `scripts/run_all.ps1` on the GPU machine.

## 1. Speed scaling (random weights, batch 1, CPU)

| Length | Student (k=16) s | Full-attention transformer (MiniLM-L6 shape) s |
|---|---|---|
| 128 | 0.025 | 0.019 |
| 256 | 0.048 | 0.058 |
| 512 | 0.092 | 0.161 |
| 1024 | 0.195 | 0.522 |
| 2048 | 0.375 | 1.948 |
| 4096 | 0.737 | 8.021 |

Fitted exponent: student latency ~ length^0.98, transformer ~ length^1.73. Student is slower at very short inputs (sequential chunk loop) and about 11x faster at 4096 tokens.

## 2. Synthetic bag-of-words distillation

The teacher is the normalized mean of fixed random word vectors (100-word vocabulary, 64-d). Matching it requires carrying every chunk's contribution in one vector. Single seed, 1 read layer, 1500 steps, batch 64.

Trained on texts of 4-64 words (cosine to teacher):

| Chunk size k | Train time | In-distribution | 128 words | 256 words | 512 words |
|---|---|---|---|---|---|
| 16 | 35 s | 0.943 | 0.949 | 0.920 | 0.823 |
| 4 | 72 s | 0.950 | 0.947 | 0.886 | 0.788 |
| 1 | 204 s | 0.992 | 0.977 | 0.901 | 0.827 |

Trained on texts of 4-256 words, k=16: 0.954 at 256, 0.964 at 512, 0.915 at 1024.

## What this does and does not show

- The read-write cell learns, and its vector carries information across dozens of chunks.
- Chunking costs some precision (k=1 is best in-distribution) but trains 6x faster than k=1, and k=16 matched k=4.
- The quality drop on much longer inputs is mostly extrapolation: training on longer texts fixes it.
- Not shown: real-language quality. This task is order-insensitive, so it does not test whether reading conditioned on the vector helps with meaning. Single seed, small model, easy teacher.
