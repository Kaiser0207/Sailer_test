# SAILER Paper vs. Code - Full Compliance Audit (V2)

**Audit Date**: 2026-03-29
**Paper**: "Developing a Top-tier Framework in Naturalistic Conditions Challenge for Categorized Emotion Prediction"
**Codebase**: `/home/brant/Project/SAILER_test/`

This report systematically cross-references every methodology detail in the paper against the actual implementation, identifying matches, deviations, and missing items.

---

## Section 2.1: Speech Foundation Model

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| Whisper-Large-V3 as speech model | PASS | [train.py:53-55](file:///home/brant/Project/SAILER_test/train.py#L53-L55) | `WhisperModel.from_pretrained("openai/whisper-large-v3").encoder` |
| Use only Encoder (not full model) | PASS | [train.py:55](file:///home/brant/Project/SAILER_test/train.py#L55) | `.encoder` correctly strips the Decoder |
| **Use only last hidden layer output** (not weighted average) for Whisper | **MISMATCH** | [train.py:154](file:///home/brant/Project/SAILER_test/train.py#L154) | Paper states: *"for Whisper Large-V3, we use only the representations from the last layer."* Code uses `.last_hidden_state` which IS the last layer. **This is CORRECT.** |
| 3-layer pointwise Conv (filter=256) | PASS | [sailer_model.py:39-48](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L39-L48) | Three `Conv1d(kernel_size=1)` layers, hidden_dim=256 |
| ReLU activation between conv layers | PASS | [sailer_model.py:41,44,47](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L41) | `nn.ReLU()` after each conv |
| Temporal averaging after conv | PASS | [sailer_model.py:129](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L129) | Vectorized masked mean pooling |
| Freeze Whisper (fine-tune only downstream) | PASS | [train.py:60-63](file:///home/brant/Project/SAILER_test/train.py#L60-L63) | `requires_grad = False` for all Whisper params |
| Max speech input = 15 seconds | PASS | [msp_dataset.py:282](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L282) | `target_frames = 1500` (Whisper: 100 frames/sec -> 1500 = 15s) |

---

## Section 2.1: Text Model

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| RoBERTa-Large as text model | PASS | [train.py:50,57](file:///home/brant/Project/SAILER_test/train.py#L50) | `roberta-large` |
| Weighted average of ALL encoder outputs | PASS | [sailer_model.py:33-35,137-143](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L33-L35) | 25-layer learnable weighted average with softmax normalization |
| Temporal averaging on text | PASS | [sailer_model.py:146](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L146) | Masked mean pooling using `t_mask` |
| Concatenation of speech + text | PASS | [sailer_model.py:155](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L155) | `torch.cat([s_emb, t_emb], dim=1)` |
| 2-layer MLP with ReLU for classification | PASS | [sailer_model.py:57-62](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L57-L62) | `Linear -> ReLU -> Dropout -> Linear` |

---

## Section 2.2: Learning Objective

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| KL Divergence Loss (not CrossEntropy) | PASS | [train.py:95-96](file:///home/brant/Project/SAILER_test/train.py#L95-L96) | `nn.KLDivLoss(reduction='batchmean')` |
| Soft labeling (vote distribution) | PASS | [msp_dataset.py:73-118](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L73-L118) | Aggregates raw annotator votes into probability distributions |
| Include "No Agreement" samples in training | PASS | [msp_dataset.py:129-138](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L129-L138) | All Train samples included regardless of consensus status |

---

## Section 2.3: Data Augmentation

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| Annotation Dropout: drop 20% of annotations | PASS | [msp_dataset.py:200-207](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L200-L207) | `n_drop = max(1, int(total_votes * 0.2))`, drops from majority classes |
| Drop specifically from majority classes | PASS | [msp_dataset.py:203](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L203) | `drop_pool = [i for i in self.majority_classes if v[i] > 0]` |
| Audio Mixing: mix majority with minority | PASS | [msp_dataset.py:241-242](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L241-L242) | Triggered when `consensus_label in majority_classes` |
| Minority sample selected by inverse distribution | PASS | [msp_dataset.py:57-69,242](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L57-L69) | `random.choices(..., weights=self.record_weights)` using inverse frequency |
| Random order of xmaj and xmin | PASS | [msp_dataset.py:248-253](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L248-L253) | 50/50 coin flip for ordering |
| Coin flip: silence vs overlap | PASS | [msp_dataset.py:255](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L255) | `random.choice(["silence", "overlap"])` |
| **Sample time t in [0, 2] seconds** | **DEVIATION** | [msp_dataset.py:258,263](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L258) | Paper: *"sample a time value t in [0, 2]"*. Code: `random.randint(0, 100)` frames. At 100 frames/sec, this is [0, 1] seconds, not [0, 2]. **Should be `random.randint(0, 200)`.** |
| Mixed distribution: `d_mix = (d_maj + d_min) / 2` | PASS | [msp_dataset.py:278-280](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L278-L280) | `(label_dist + min_dist) / 2.0` |

---

## Section 2.4: Engineering Design Choices

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| Distribution Re-weighting: `w_i = 1/q_i` | PASS | [msp_dataset.py:50-55](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L50-L55) | `w = 1.0 / (q + 1e-8)`, then normalized |
| Re-weighting applied element-wise: `d' = d * w` | PASS | [msp_dataset.py:212](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L212) | `d_prime = d * self.w_norm` |
| Re-weighting NOT applied during validation | PASS | [msp_dataset.py:214-215](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L214-L215) | `else: d_prime_normalized = d` |
| Minority class mAP as validation metric | PASS | [train.py:246-255](file:///home/brant/Project/SAILER_test/train.py#L246-L255) | Computes per-class AP for Fear, Disgust, Surprise, Contempt |
| Include Dev 'other'/'no-agreement' in training | PASS | [msp_dataset.py:133-138](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L133-L138) | Fetches Development set unagreed samples |

---

## Section 3.2: Experimental Details

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| Learning rate = 0.0005 | PASS | [default_config.json](file:///home/brant/Project/SAILER_test/configs/default_config.json) | `"learning_rate": 0.0005` |
| Epochs = 15 | PASS | [default_config.json](file:///home/brant/Project/SAILER_test/configs/default_config.json) | `"epochs": 15` |
| Filter size (hidden_dim) = 256 | PASS | [sailer_model.py:20](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L20) | `hidden_dim=256` |
| Max speech = 15 seconds | PASS | [msp_dataset.py:282](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L282) | `target_frames = 1500` |
| Fixed seed | PASS | [train.py:38](file:///home/brant/Project/SAILER_test/train.py#L38) | `set_seed(42)` |
| **Whisper: use LAST layer only** | PASS | [train.py:154](file:///home/brant/Project/SAILER_test/train.py#L154) | `.last_hidden_state` is exactly the last encoder layer |
| **RoBERTa: weighted average of ALL layers** | PASS | [sailer_model.py:137-143](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L137-L143) | 25-layer weighted average |
| Multi-task: primary + secondary + AVD | PASS | [train.py:164-170](file:///home/brant/Project/SAILER_test/train.py#L164-L170) | All three losses computed and summed |

---

## Section 4.3: Predicting Additional Labels

| Paper Detail | Status | Code Location | Notes |
|---|---|---|---|
| Secondary emotion prediction (17 classes) | PASS | [sailer_model.py:66-71](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L66-L71) | Separate MLP head |
| Arousal/Valence/Dominance regression | PASS | [sailer_model.py:74-95](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L74-L95) | Three separate MLP heads with Sigmoid |

---

## CRITICAL FINDINGS

> [!CAUTION]
> **Finding 1: Audio Mixing Time Range is Wrong**
> - **Paper**: *"we sample a time value t in [0, 2] to determine the duration of the silence or overlap"*
> - **Code**: `random.randint(0, 100)` frames = [0, 1] second
> - **Impact**: The augmentation variability is halved. Silence/overlap segments should range up to 2 seconds (200 frames), not 1 second (100 frames).
> - **Fix**: Change `random.randint(0, 100)` to `random.randint(0, 200)` in both the silence and overlap branches (lines 258 and 263).

> [!WARNING]
> **Finding 2: Annotation Dropout runs unconditionally (Train AND Validation)**
> - **Paper**: Annotation Dropout is a *training-time* data augmentation technique.
> - **Code**: `_get_target_distribution()` always executes the dropout logic (lines 200-207) regardless of whether `is_training` is True or False. The `is_training` parameter is received but never checked.
> - **Impact**: Validation labels are being randomly perturbed every epoch, causing noisy and unreliable validation metrics. This partially explains the zig-zag `val_macro_f1` curve observed in V1.
> - **Fix**: Wrap the annotation dropout block in `if is_training:`.

> [!WARNING]
> **Finding 3: `dropout_rate` config value is not being applied to the model**
> - **Config**: `"dropout_rate": 0.3`
> - **Code**: [train.py:82](file:///home/brant/Project/SAILER_test/train.py#L82) hardcodes `dropout_rate=0.2`
> - **Impact**: The config file's dropout_rate is silently ignored. The model always uses 0.2.
> - **Fix**: Change to `dropout_rate=config.get("dropout_rate", 0.2)`.

> [!NOTE]
> **Finding 4: `logger` variable is undefined after refactoring**
> - **Code**: [train.py:141](file:///home/brant/Project/SAILER_test/train.py#L141) and [train.py:265](file:///home/brant/Project/SAILER_test/train.py#L265) reference `logger` but it was never assigned. Should be `tracker.logger`.
> - **Impact**: This will crash at runtime when Epoch 1 starts.
> - **Fix**: Replace `logger.info(...)` with `tracker.logger.info(...)`.

> [!NOTE]
> **Finding 5 (Non-Paper): L2 Normalization is an addition not in the original paper**
> - The paper does NOT mention L2 normalization before concatenation.
> - This is our own engineering improvement. It may help or hurt; monitor V2 results carefully.
> - If results degrade, consider removing it.

## SUMMARY

| Category | Total | Pass | Deviation | Missing |
|---|---|---|---|---|
| Model Architecture | 10 | 10 | 0 | 0 |
| Learning Objective | 3 | 3 | 0 | 0 |
| Data Augmentation | 7 | 6 | **1** | 0 |
| Engineering Choices | 5 | 5 | 0 | 0 |
| Experimental Details | 8 | 8 | 0 | 0 |
| Multi-task | 2 | 2 | 0 | 0 |
| **Code Bugs** | 3 | 0 | **3** | 0 |

**Verdict**: The core methodology faithfully reproduces the paper. Three code-level bugs (annotation dropout in validation, audio mixing time range, undefined logger) require immediate fixes.
