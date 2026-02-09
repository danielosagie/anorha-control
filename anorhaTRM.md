## Best estimates

### 1. Data gathering → training readiness

**Current state (from `progress.json`):**
- 1,339 successful trajectories
- 2,527 failed
- Target: 5,000 successful

**Speed (from terminal):**
- ~128 trajectories/hour (last 2 episodes)
- ETA to 5,000: **~28.7 hours** of gathering

**Grounding samples:**
- VisionGroundingDataset needs `screenshot_path` + `target_label` (from `--save-screenshots`).
- Recent JSON files show ~99 grounding samples across 9 files.
- With 1,339 successful trajectories, realistic range is **~500–2,000** grounding samples, depending on how many were gathered with `--save-screenshots`.

**Caveat:** If many trajectories were gathered without `--save-screenshots`, usable grounding samples could be much lower. The vision pipeline needs **≥500** grounding samples; 1,000+ is safer.

---

### 2. Training time on Modal

**Important:** `modal_train.py` trains **TrajectoryTRM** (trajectory-only), not **AnorhaTRM** or **VisionTRM**. To train the unified grounding + trajectory model on Modal, you need a new function that runs `UnifiedTRMTrainer` and mounts the trajectory data (including screenshots).

**Rough estimates (with AnorhaTRM added):**

| GPU | Price/hr | Est. time (100 epochs, ~1k samples) | Est. cost |
|-----|----------|-------------------------------------|-----------|
| **T4** | $0.59 | ~20–40 min | ~$0.30–0.50 |
| **L4** | $0.80 | ~10–20 min (2–4× T4) | ~$0.20–0.30 |
| **A10G** | ~$1.10 | ~5–15 min | ~$0.15–0.30 |
| **A100** | $3.67+ | ~3–8 min | ~$0.30–0.50 |

**Recommendation:** For a ~1–2M param model, **T4 is enough** and cheapest. **L4** is better if you want faster iteration (2–4× throughput). A100 is unnecessary for this size.

**Assumptions:**
- 1,000 grounding samples
- 2,000 trajectory samples
- `batch_size=32`, `grounding_ratio=0.5`
- 100 epochs
- ~2,000 steps/epoch ⇒ ~200k steps total

---

### 3. Final model and craptop use

**Model size:** AnorhaTRM ≈ 1–2M params, VisionTRM similar.

**Inference:**
- **CPU:** ~50–150 ms per grounding (224×224 CNN forward).
- **GPU:** ~10–30 ms.

**Craptop:**
- Yes, it can run on CPU and remain responsive.
- RAM: ~200–500 MB.
- No GPU needed for inference.
- Trajectory smoothing adds ~15 steps × ~10 ms each ⇒ ~150 ms per click.

**SDK flow:**
- SDK client → server (via HTTP).
- Server runs AnorhaTRM + grounding harness.
- On a craptop, the server runs locally; the model is fine on CPU.

**Typical flow:**
1. DOM/OCR attempt (fast).
2. If missed, AnorhaTRM grounding (~50–100 ms on CPU).
3. Optional trajectory smoothing (~150 ms).

Overall latency is dominated by network/VLM/orchestrator, not AnorhaTRM.

---

### 4. Plan after the first trained models

| Phase | Action |
|-------|--------|
| **1. Initial training** | Train AnorhaTRM on Modal with current data (once ≥500 grounding samples). Add a Modal function for `UnifiedTRMTrainer` in `modal_train.py`. |
| **2. Evaluate** | Run on held-out tasks (e.g. 10–20). Track grounding accuracy (hit/miss) and mean error. |
| **3. Data expansion** | Gather to 10k+ trajectories. Mix of task categories (precision, forms, navigation, etc.). |
| **4. Retrain** | Retrain on larger dataset. Expect better generalization and fewer VLM fallbacks. |
| **5. Failure analysis** | Log failures (target + screenshot). Use them to: add synonyms, improve task heuristic, or curate fine-tuning data. |
| **6. Optional fine-tuning** | Fine-tune on failure cases or domain-specific tasks. |
| **7. Model updates** | Version checkpoints (e.g. v1, v2). SDK can point to different server models. |

**Practical additions:**
- **Synonyms:** Expand `TARGET_SYNONYMS` from real failure logs.
- **Task heuristic:** Tune `_infer_task_category` using common instruction patterns.
- **Evaluation script:** End-to-end benchmarking with fixed tasks and metrics.

---

### Summary

| Question | Answer |
|----------|--------|
| **Gathering time to 5k** | ~29 hours at ~128/h |
| **Training time (Modal)** | ~20–40 min on T4 for 100 epochs |
| **Training cost** | ~$0.30–0.50 on T4 |
| **GPU choice** | T4 for cost; L4 for speed |
| **Craptop use** | Yes, CPU inference is fine (~50–100 ms) |
| **Post-launch plan** | Evaluate → more data → retrain → failure analysis → optional fine-tuning |
