# Read-Write Encoder: O(n) Text Embedding Model — Design Spec

Date: 2026-09-29
Status: Draft for review

## 1. Goal

Train a text embedding model that runs in O(n) time in the input length. The model reads the text in one pass and maintains a single fixed-size vector. The current vector conditions how upcoming tokens are interpreted, and the model then edits the vector. The final vector is the embedding.

Training plan: distill from a small pretrained embedder first, then fine-tune contrastively to try to exceed the teacher.

## 2. Decisions

| Topic | Decision |
|---|---|
| Objective | Stage 1: distillation from a teacher. Stage 2: contrastive fine-tuning (designed after Stage 1 results). |
| Hardware | Local NVIDIA GPU, 6-8 GB VRAM, Windows. Pure PyTorch, no custom CUDA kernels. |
| v1 success | Reach about 90-95% of the teacher's Spearman on STS-B and STS12-16, plus report cosine agreement with the teacher. |
| Architecture | Chunked read-write cell (chunk size k, default 16). k=1 is the per-token ablation. |
| Teacher | `all-MiniLM-L6-v2` (384-d, 22M parameters) |
| Tokenizer | Teacher's WordPiece tokenizer. Token embeddings initialized from the teacher. |

## 3. Model

State: one vector `v` of 384 dimensions, initialized from a learned `v0`.

Position is encoded within a chunk only (0 to k-1). There is no global position embedding, so the model is not tied to a maximum length.

One step per chunk, with weights shared across all chunks:

1. **Read.** The k tokens of the chunk pass through a few small transformer layers. Each token attends to the other tokens of the chunk and to the current `v`, which is added as an extra key/value slot. So `v` conditions the interpretation of every token in the chunk.
2. **Summarize.** `v` acts as the query in a cross-attention over the processed chunk tokens, producing a chunk summary `s`.
3. **Write.** `delta = MLP([v, s])`, `g = sigmoid(W [v, s])`, `v <- LayerNorm(v + g * delta)`.
4. **Readout.** After the last chunk, a linear head on `v` gives the embedding. The same head is applied to intermediate `v` values for the auxiliary loss.

Padding: the final chunk is padded. Pad tokens are masked in attention. A chunk that is entirely padding has its gate forced to 0, so `v` is unchanged.

Defaults:

| Setting | Default |
|---|---|
| Vector size | 384 |
| Chunk size k | 16 (ablate 1, 4, 16, 32) |
| Read layers | 2 initially, up to about 6 if VRAM allows |
| Max training length | 256 tokens (16 sequential steps) |
| Parameters | about 16M (2 read layers), about 25M (6 read layers) |

Complexity: O(n*k) time with k fixed, so linear in n. Inference memory is constant because only `v` is carried between chunks. Text can be streamed, and an embedding can be read out at any chunk boundary.

## 4. Data and teacher

The teacher is run once over the whole training set. Its vectors are stored as fp16 (about 770 MB per 1M texts). Training never runs the teacher.

Texts are capped at 256 tokens. Most texts should be 128 tokens or fewer because the teacher is mostly trained on that range. Longer texts are kept but rarer.

| Source | Purpose |
|---|---|
| Wikipedia passages | Broad topics, medium and long lengths |
| AllNLI sentences | Short, similarity-flavored sentences |
| MS MARCO queries and passages | Search-style text |
| STS-B train split | Domain match with the eval. Dev and test splits are never used for training. |

Start with about 1M texts. Batches are bucketed by length to reduce padding.

## 5. Training (Stage 1: distillation)

Losses:

| Loss | Definition | Weight |
|---|---|---|
| Cosine to teacher | `1 - cos(student_final, teacher)` | 1.0 |
| Similarity-matrix match | Student's batch-by-batch cosine matrix matched to the teacher's | 1.0 |
| Intermediate readouts | Vectors after each chunk pulled toward the teacher's full-text vector, more weight on later chunks | 0.2 |

Loop:

- AdamW, peak learning rate about 5e-4, 1k warmup steps, cosine decay, gradient clipping at 1.0.
- Mixed precision selected automatically: bf16 if supported, else fp16 with loss scaling.
- Batch size 128, with gradient accumulation if VRAM is tight.
- Regular checkpoints, resumable. Every N steps: STS-B dev Spearman and teacher cosine agreement, logged to CSV.

## 6. Stage 2 (contrastive)

Contrastive fine-tuning on AllNLI triplets with in-batch negatives, keeping a small distillation loss as a regularizer. The detailed design is written after Stage 1 results exist.

## 7. Evaluation

| Check | Measures |
|---|---|
| STS-B dev/test, STS12-16 | Cosine-similarity Spearman versus the teacher's score. This is the v1 success number. |
| Teacher agreement | Mean cosine between student and teacher on held-out text |
| Scaling benchmark | Latency and peak memory at 128, 512, 2k, 8k tokens for student and teacher, showing linear growth for the student |
| Long-input sanity | Embed concatenated texts to check quality beyond the training length |
| k ablation | Chunk sizes 1, 4, 16, 32: quality versus training time |

## 8. Repository layout and environment

```text
Mamba Vision Testing/
  ORGANIZATION.md
  README.md
  requirements.txt
  configs/          base.yaml, k1_ablation.yaml
  docs/specs/       design specs
  src/rwe/          model.py, data.py, losses.py, train.py,
                    teacher_cache.py, eval_sts.py, benchmark_scaling.py
  scripts/          setup and data-prep entry points
  tests/
  data/             raw text and cached teacher vectors (not in git)
  runs/             checkpoints and logs (not in git)
```

Environment: Python venv on Windows with CUDA PyTorch. The user runs training commands in PowerShell. `ORGANIZATION.md` is kept current as the layout evolves.

## 9. Validation without a GPU

CPU-only tests run in the development environment:

- Output shapes, and gradients reaching every parameter.
- Padding invariance: adding pad tokens does not change the embedding.
- Streaming equivalence: feeding chunks one at a time gives the same vector as a batch pass.
- k=1 behaves as the per-token cell.
- Tiny overfit run on about 64 texts, where the loss must fall.
- Smoke run of the full training loop with a tiny config.

Real GPU speed and final quality can only be measured on the user's machine. Each real run is a single command.

## 10. Milestones

| # | Milestone |
|---|---|
| M0 | Scaffold, model and tests passing on CPU |
| M1 | Data prep and teacher vector cache |
| M2 | First training run and STS numbers |
| M3 | Chunk-size ablation and scaling benchmark |
| M4 | Contrastive stage |

## 11. Risks

- A single vector may not hold enough information for long texts. Mitigation: two-pass read (feed the text twice, read out after the second pass), still O(n).
- Gradients through sequential steps may be unstable. Mitigation: gated residual edits and the LayerNorm on `v`.
- Quality is capped by the teacher until Stage 2.
- Teacher quality degrades on texts longer than about 128 tokens, so those targets are noisier.

## 12. Non-goals

- Custom CUDA kernels or Mamba/SSM layers (possible later swap, out of scope for v1).
- Training a new tokenizer.
- Multilingual support.
