# Long-Memory Frame Selection — Working Notes

Research notes for the attention-guided long-memory frame-selection work on LSTR
(THUMOS'14). Living document — update as experiments run.

## Motivation

LSTR compresses a long-memory window (hundreds–thousands of past frames) into a
few tokens via stage-1 cross-attention. Much of that long memory is background /
irrelevant. **Hypothesis:** most long-memory frames are noise, so selecting only
the relevant ones should preserve (or improve) accuracy — and ideally cut compute.

Progression of the idea:
1. **Oracle mask** — use ground-truth labels to keep only action frames. Upper-bound
   check: "does removing background long-memory frames help?" (branch
   `feat/oracle-experiment`).
2. **Attention-guided selection** — replace the oracle with the model's *own*
   stage-1 attention as the frame-importance score (no labels, no new weights).
   → **this branch: `feat/attention-frame-selection-singlepass`**.
3. **Learned pre-embedding gate** (planned) — the only path to real speedup.

---

## Key facts (THUMOS'14, this repo)

- **fps = 4**, 22 classes (incl. Background=0, Ambiguous=21), 413 videos
  (200 train/val + 213 test).
- Long-memory frame count: `N = LONG_MEMORY_SECONDS × fps ÷ LONG_MEMORY_SAMPLE_RATE`.
  - e.g. `SECONDS=2048, RATE=4` → **N = 2048**;  `SECONDS=512, RATE=4` → **N = 512**.
- Model size: **58.9M params** (235 MB fp32).
- Forward cost: **24.66 GFLOPs / window** at N=2048 (see breakdown below).

---

## What's implemented

### 1. Config block — `config/defaults.py`
```yaml
MODEL: { LSTR: { FRAME_SELECTION: {
  ENABLED: False,   # off by default → baseline forward is byte-for-byte unchanged
  TOP_K:   128,     # absolute number of long-memory frames to KEEP
  MODE:    'drop',  # 'drop' = discard non-selected ('fuse' not yet implemented)
} } } }
```
Selection is **inference-compatible**: it adds **no trainable parameters**, so a
baseline checkpoint runs unchanged with selection toggled on.

### 2. Attention-guided selection — single-pass, gather (current branch)
In stage-1 cross-attention (`models/transformer/multihead_attention.py`):
- 16 query tokens attend over all N long-memory frames.
- Each frame is scored by the attention it receives (mean over heads + queries).
- Top-k frames are kept; the stage-1 output is built from **only those k**.

**Implementation = "Upgrade A" (gather + deferred V-projection):**
- Project Q and **all** K (K needed to score every frame).
- Select top-k, then **value-project and aggregate only those k frames**.
- Output is mathematically identical to masking-then-renormalizing
  (verified max diff ~3e-16), so **mAP is unchanged** — but the V-projection and
  value matmul now scale with `k`, so stage-1 FLOPs actually drop.

### 3. Two-pass variant — branch `feat/attention-frame-selection`
Score with one full stage-1 pass, then re-compress over the selected frames in a
second pass. Same result as single-pass, but ~2× stage-1 cost. Superseded by
single-pass; kept for comparison.

### 4. Oracle mask — branch `feat/oracle-experiment`
Ground-truth masking of background long-memory frames (`all_actions` /
`match_class` modes). Upper-bound experiment.

### 5. Tooling
- **`tools/flop_frame_selection.py`** — counts matmul FLOPs (F.linear + bmm) for
  one forward, per-stage breakdown, baseline vs TOP_K sweep, plus a training-run
  total estimate (`--epochs/--train_iters/--test_iters`). No GPU/checkpoint needed.
- **`utils/flops.py` + trainer wiring** — logs params, forward GFLOPs/window, and
  estimated training PFLOPs at the **start of every training run**.
- **Confirmation log** — on first forward, prints
  `single-pass frame selection ACTIVE: keeping K / N` (or a no-op warning when
  `TOP_K ≥ N`), so runs are self-verifying.
- **Non-finite guard** in the trainer — detects NaN/inf model outputs, logs them,
  skips poisoned steps (was hit earlier as `det_loss=0.00000` + NaN divergence).
- **`tools/smoke_frame_selection.py`** — asserts single-pass == two-pass, k≥N is a
  no-op, padded frames excluded, etc.

### 6. Configs & jobs
- `configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_frameselect_singlepass.yaml`
- **PBS run scripts live in the separate `hpc_run` repo** (`run_job/run_lstr.pbs`),
  not in this repo. `run_lstr.pbs` trains one baseline, then evals the same
  checkpoint with selection OFF then ON (clean head-to-head mAP), runs the FLOP
  sweep, and includes the DataLoader OOM overrides.

---

## Key findings

### FLOP breakdown (per window, N=2048)
| Stage | GFLOPs | Share | Prunable by attention selection? |
|---|---|---|---|
| long_feature_head | 12.885 | 52% | ❌ embeds all frames *before* selection |
| stage1_encoder | 8.994 | 36% | partly — V-proj + value matmul only |
| stage2_encoder | 1.221 | 5% | ❌ operates on 16 tokens |
| decoder | 1.359 | 6% | ❌ |
| work_feature_head | 0.201 | 1% | ❌ |
| **TOTAL** | **24.661** | | |

### The "+0.0% → ~15%" story
- The **original masking** implementation zeroed weights but kept every matmul
  full-size → **+0.0% FLOP change** (selection changed the *output* but not the
  *compute*).
- **Upgrade A (gather)** makes the V-projection + value matmul scale with k →
  **~15% total reduction** at k=128 (stage1: 8.99 → ~5 GFLOP).
- **Ceiling ≈ 15%** for attention-based selection: the feature head (52%) and the
  K-projection must run over all frames, because attention needs every frame's
  embedding to rank it.

### Training result (job 24927529, 2026-08-12)
- Trained **with** selection (TOP_K=128, N=2048 → keeps ~6%), stable, no NaN.
- Final perframe mAP ≈ **0.7055** (peak 0.708 @ epoch 18).
- ⚠️ `test_net` OOM-crashed (DataLoader) — fixed since (BATCH_SIZE/NUM_WORKERS/
  PIN_MEMORY overrides). Canonical batch-inference mAP still to be captured.
- ⚠️ **No baseline (selection-OFF) number yet** → can't yet quantify the accuracy
  cost/benefit of selection. The `hpc_run` baseline-vs-selection job addresses this.

---

## What's planned

### Upgrade B — prune BEFORE the feature head (the real speedup) 🔴
Score frames with a **cheap proxy** (feature/motion magnitude, or a tiny learned
`Linear(→1)` gate) *before* embedding, so the feature head + stage-1 process only k
frames.
- Potential saving: **up to ~80%** (cuts the dominant 52% feature head).
- Trade-off: cheaper score ⇒ worse selection ⇒ accuracy risk to study.
- This is the intended headline contribution (a lightweight learned frame gate).

### `fuse` mode (EViT-style)
Instead of discarding non-selected frames, merge them into one score-weighted
"leftover" token so their information isn't fully lost. Currently
`NotImplementedError`.

### Stream-inference support
Selection is wired only into the **batch** forward path (`LSTR.forward`);
`stream_inference` is untouched. Needed for online/streaming eval.

### Experiments to run
- [ ] Baseline vs selection mAP (hpc_run: run_job/run_lstr.pbs).
- [ ] Accuracy-vs-k sweep (TOP_K ∈ {64,128,256,512}) → find the knee.
- [ ] Confirm Upgrade A shows ~15% FLOP drop in the FLOP script output.
- [ ] Segmentation metrics (edit score, F1@k) alongside mAP — richer than mAP for
      action/background transitions (prototype was on `feat/segment-based`).

---

## How to run

**Train baseline, then eval OFF vs ON (clean comparison)** — run from the `hpc_run` repo:
```bash
cd ~/hpc_run/run_job && ./submit.sh run_lstr.pbs
```

**FLOPs (no GPU/checkpoint needed):**
```bash
python tools/flop_frame_selection.py \
  --config_file configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x_frameselect_singlepass.yaml \
  --top_k 64 128 256
```

**Toggle selection via CLI overrides (no config edit):**
```bash
python tools/test_net.py --config_file <cfg> --gpu 0 \
  MODEL.CHECKPOINT <ckpt.pth> \
  MODEL.LSTR.FRAME_SELECTION.ENABLED True \
  MODEL.LSTR.FRAME_SELECTION.TOP_K 128 \
  DATA_LOADER.BATCH_SIZE 4 DATA_LOADER.NUM_WORKERS 4 DATA_LOADER.PIN_MEMORY False
```

---

## Branch map
| Branch | Contents |
|---|---|
| `feat/attention-frame-selection-singlepass` | **current** — single-pass gather selection (Upgrade A) + tooling |
| `feat/attention-frame-selection` | two-pass selection variant |
| `feat/oracle-experiment` | ground-truth oracle mask |
| `feat/segment-based` | segmentation metrics (edit / F1@k) |
| `feat/random-masking`, `feat/selectable-masking`, `feat/vla-pruner` | earlier / exploratory |
| `main` | upstream baseline |

## Gotchas
- **`TOP_K ≥ N` is a no-op** (keeps all frames). With fps=4, check
  `N = SECONDS × 4 ÷ RATE` before setting TOP_K. The confirmation log will say if
  it's a no-op.
- **Config file ≠ algorithm.** Single-pass vs two-pass is decided by the **branch
  code**, not the `.yaml`; the two frameselect configs differ only in comments.
- **FLOPs ≠ VRAM ≠ time.** FLOP script measures arithmetic only (random weights,
  fake input). Batch-inference OOM was system-RAM, fixed via loader overrides.
