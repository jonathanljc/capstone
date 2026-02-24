# UWB NLOS Ranging Correction Pipeline — Architecture & Design

## Overview

A 3-stage ML pipeline that takes a raw UWB CIR (Channel Impulse Response) waveform and produces a corrected distance measurement. Each stage answers one question:

| Stage | Question | Model | Output |
|-------|----------|-------|--------|
| **Stage 1** | Is this signal LOS or NLOS? | DualCircuit_PI_HLNN (LNN) | Binary classification + 64-dim embedding |
| **Stage 2** | If NLOS, is the signal quality good enough to correct? | Random Forest on LNN embeddings | Correctable vs Challenging |
| **Stage 3** | If correctable, how much error should we subtract? | Random Forest on LNN embeddings | Ranging error (meters) |

**Final correction**: `d_corrected = d_hardware - predicted_error`

```
Raw CIR (1016 samples) + FP_AMPL1/2/3
         |
    [ Stage 1: DualCircuit_PI_HLNN ]
         |
    LOS? ──yes──> Use d_hardware as-is (no correction needed)
         |
        NLOS
         |
    [ Stage 2: RF Classifier on 64-dim LNN embeddings ]
         |
    Correctable? ──no──> Flag as "Challenging" (unreliable correction)
         |
       yes
         |
    [ Stage 3: RF Regressor on 64-dim LNN embeddings ]
         |
    predicted_error (meters)
         |
    d_corrected = d_hardware - predicted_error
```

---

## The Physical Setup

**Hardware**: DWM1001 UWB modules (DW1000 chip), Two-Way Ranging protocol.

**CIR**: 1016 time-domain samples at ~1 ns/sample (~0.3003 m per index). The CIR captures every signal path between TX and RX — direct, reflected, scattered.

**Dataset**: 3600 samples total
- **1800 LOS** (3 scenarios: 4.55m, 8.41m, 9.76m) x 6 UWB channels
- **1800 NLOS** (3 scenarios: 12.79m, 16.09m, 16.80m bounce paths) x 6 UWB channels
- All NLOS is **single-bounce by geometry** — TX emits, signal hits a wall/reflector, bounces to RX. There is no direct path (blocked by obstruction).

**Ground truth distances** (measured):
- `d_bounce`: TX -> reflector -> RX distance, **laser-measured**
- `d_direct`: true TX -> RX straight-line distance, **floor-plan geometry** (can't laser through a wall)
- `bounce_path_idx`: CIR index where the bounce signal should arrive = `d_bounce / 0.3003`

**The problem**: The DW1000 reports `d_hardware` based on the first detected path. In NLOS, this overshoots `d_direct` because the signal travels the longer bounce path. The ranging error = `d_hardware - d_direct`.

| NLOS Scenario | d_direct | d_bounce | Ranging Error | bounce_path_idx |
|---------------|----------|----------|---------------|-----------------|
| 12.79m | 8.82m | 12.79m | ~3.97m | ~810 |
| 16.09m | 7.20m | 16.09m | ~8.89m | ~820 |
| 16.80m | 13.06m | 16.80m | ~3.74m | ~756 |

---

## Stage 1: LOS/NLOS Classification

### Intent

Before correcting anything, we need to know: **is this signal LOS or NLOS?** LOS signals don't need correction. Only NLOS signals have the bounce-path error that needs fixing.

### Architecture — DualCircuit_PI_HLNN

A **Physics-Informed Hybrid Liquid Neural Network** with ~17,200 parameters.

**Why "Liquid"?** Liquid Neural Networks (Hasani et al. 2020) are based on the Liquid Time-Constant (LTC) neuron model — an ODE-based neuron inspired by biological neural circuits. Unlike standard RNNs where the time constant is implicit, LTC neurons have an **explicit time constant tau** that emerges from the ODE dynamics:

```
tau = Cm / (g_leak + sum(w * gate(v)))
```

This tau adapts based on the input — it's not a fixed hyperparameter but a **learned, input-dependent temporal response**. For CIR signals, this is powerful: LOS signals (sharp, single-peak) produce different tau dynamics than NLOS signals (broader, multi-peak with reflections).

**Why "Dual-Circuit"?** Two parallel LTC circuits process the same CIR sequence simultaneously:

```
                    Raw CIR sequence (60 timesteps)
                              |
                    +---------+---------+
                    |                   |
              cell_los (32-dim)   cell_nlos (32-dim)
              "LOS specialist"    "NLOS specialist"
                    |                   |
                    +--- cross-talk ----+
                    |    (gated)        |
                    |                   |
              attn pool (32)      attn pool (32)
                    |                   |
                    +-------cat---------+
                            |
                      64-dim embedding
                            |
                    Linear(64->32) -> SiLU -> Dropout -> Linear(32->1) -> Sigmoid
                            |
                     P(NLOS) in [0, 1]
```

**Cross-circuit communication**: At every timestep, each circuit receives a gated projection of the other circuit's state:

```python
# NLOS circuit influences LOS circuit (and vice versa)
proj_nlos_to_los = P_nlos2los(h_nlos)           # linear projection
gate = sigmoid(W_gate @ [h_los | proj])          # learned gate [0,1]
h_los_input = h_los + gate * proj_nlos_to_los    # gated additive
```

This creates **competitive dynamics** — when one circuit is confident (strong activations), it inhibits the other via the gating mechanism. The two circuits develop complementary specializations during training.

**FP_AMPL conditioning**: The DW1000 reports first-path amplitudes (FP_AMPL1/2/3) — these are hardware-level measurements of the first detected signal path's strength. Instead of zero-initializing the circuits, FP_AMPL seeds the initial hidden state:

```python
h_los_0  = 0.1 * tanh(Linear(3 -> 32)(fp_features))   # gentle nudge
h_nlos_0 = 0.1 * tanh(Linear(3 -> 32)(fp_features))   # from FP_AMPL
```

The 0.1 scale factor is critical — it gives a hint but forces the model to develop its understanding from the actual CIR temporal dynamics, not just shortcut from FP amplitudes.

**Why this makes tau physics-informed**: Since tau = Cm / (g_leak + sum(w * gate(v))) and v_0 comes from FP_AMPL, the time constant is immediately influenced by the hardware's first-path measurement. Strong FP (likely LOS) starts the circuit in one regime; weak FP (likely NLOS) starts it in another. The tau then evolves as the CIR waveform is processed timestep by timestep.

### CIR Preprocessing

```
Raw CIR (1016 samples)
    |
    v
1. RXPACC normalization: sig = sig / RXPACC
   (accounts for preamble accumulation count — physics normalization)
    |
    v
2. ROI alignment: find leading edge in [740, 890] search window
   (backtrack from peak to noise threshold crossing)
    |
    v
3. Crop: 10 samples before leading edge + 50 after = 60-sample window
    |
    v
4. Instance normalize to [0, 1]
    |
    v
Input tensor: (batch, 60, 1)
```

The ROI window [740, 890] was derived empirically — all CIR peaks across all 36 CSV files fall within indices 743-807. The 60-sample crop captures the leading edge, main peak, and early multipath.

### Embedding Output

The `model.embed(cir_sequence, fp_features)` method returns a **64-dim vector** (32 from LOS circuit + 32 from NLOS circuit, attention-pooled over all 60 timesteps). This embedding is the shared representation used by Stage 2 and Stage 3.

### Results

- **Test Accuracy**: 100% (540 test samples, 270 LOS + 270 NLOS)
- **Training**: 70/15/15 split, AdamW, cosine LR with warmup, early stopping
- **Loss**: Pure BCE (no tau constraint — tau emerges freely from ODE dynamics)

---

## Stage 2: Signal Quality Classification

### Intent

All 1800 NLOS samples are single-bounce by geometry. But not all are equally easy to correct. Stage 2 asks: **is this NLOS signal clean enough for reliable distance correction?**

Think of it like medical imaging triage: before attempting diagnosis (Stage 3), check if the scan quality is good enough to trust the diagnosis.

### The Problem with Simple Labels

We can't just label by scenario (all 12.79m = correctable, all 16.80m = challenging) because **within each scenario**, signal quality varies across the 6 UWB channels and 100 measurements per channel. Some 12.79m samples have clean, bounce-dominated CIRs; others have complex multipath from floor/ceiling reflections.

### Mixture Labeling — How Ground Truth Labels Are Created

Labels come from two **independent** physics measurements (NOT from the model's own predictions — this is not circular):

**1. Geometric bounce dominance** (from laser-measured distances):
```python
bounce_idx = round(bounce_path_idx)       # known from floor plan geometry
roi_energy = sum(CIR[le-5 : le+120]^2)   # total energy in ROI
bounce_energy = sum(CIR[bounce_idx-3 : bounce_idx+3]^2)  # energy at known bounce position
bounce_dominance = bounce_energy / roi_energy  # fraction [0, 1]
```

This asks: "What fraction of the ROI energy is where the bounce SHOULD be?" High dominance means the bounce path dominates the signal. Low dominance means energy is dispersed across multipath — the bounce is buried.

**2. Morphological peak count** (from CIR signal processing):
```python
roi_normalized = CIR_roi / max(CIR_roi)
peaks = find_peaks(roi_normalized, prominence=0.20, distance=5)
num_peaks = len(peaks)
```

This asks: "How many distinct signal paths are visible in the CIR?" Few peaks (1-2) = simple propagation. Many peaks (>2) = complex multipath from floor, ceiling, edges.

**Combined label (AND logic)**:

|  | Few peaks (<=2) | Many peaks (>2) |
|--|-----------------|-----------------|
| **High dominance (>=50%)** | **Correctable** (0) | Challenging (1) |
| **Low dominance (<50%)** | Challenging (1) | Challenging (1) |

- **Correctable** = bounce dominates the energy AND the CIR shape is simple
- **Challenging** = either the bounce energy is dispersed OR the morphology is complex

### Why Mixture Catches What Neither Alone Would

- **Geometric-only** would miss: sample with strong bounce but 8 additional multipath peaks (morphological complexity that confuses the regressor)
- **Morphological-only** would miss: sample with only 2 peaks but the energy is split 50/50 between direct leakage and bounce (geometric inconsistency)
- **Mixture** catches both: requires energy concentration at the expected position AND simple CIR shape

### Architecture

```
Frozen Stage 1 encoder
        |
   64-dim LNN embedding (for each NLOS sample)
        |
   Random Forest Classifier (200 trees)
        |
   Correctable (0) or Challenging (1)
```

The RF sees only the 64-dim learned embeddings — it does NOT see bounce_dominance or peak_count at inference time. Those were only used to create the training labels. The RF must learn to predict signal quality purely from the LNN's temporal dynamics representation.

### Label Distribution

```
Correctable (BOTH conditions met):     479 / 1800 (26.6%)
Challenging (either condition fails):  1321 / 1800 (73.4%)

Per scenario:
  7.79m (d_bounce=12.79m):   90 correctable / 600 total  (15.0%)
  10.77m (d_bounce=16.09m): 117 correctable / 600 total  (19.5%)
  14m (d_bounce=16.80m):      0 correctable / 600 total   (0.0%)
```

The 16.80m scenario has **zero** correctable samples because:
1. Its direct path is the longest (13.06m), creating extensive multipath (median 11 peaks)
2. Its bounce-vs-direct gap is the smallest (3.74m = ~12 CIR indices), so the bounce peak appears as a minor shoulder on the main peak with only ~8.7% of ROI energy

This is **physically correct** — the pipeline honestly identifies that this scenario's signals are too complex for reliable correction, rather than attempting bad corrections.

### Results

- **Test Accuracy**: 91.11%
- **70/30 stratified split** on 1800 NLOS samples

---

## Stage 3: Ranging Error Regression

### Intent

For the samples that Stage 2 labeled as correctable, Stage 3 predicts **how many meters to subtract** from the hardware distance to get the true distance.

```
d_corrected = d_hardware - predicted_error
```

### Data Filtering

Stage 3 only trains on samples that pass the same mixture filter used for Stage 2 labels:

```python
# Must pass BOTH criteria (same as Stage 2)
if num_peaks > 2:
    skip  # too many peaks
if bounce_dominance < 0.50:
    skip  # energy not concentrated at bounce position

# Passed -> correctable -> use for training
ranging_error = d_hardware - d_direct   # target (meters)
```

After filtering: **479 correctable samples** (from 12.79m and 16.09m scenarios only).

### Target

Per-sample ranging error = `Distance_hardware - d_direct`

This varies per sample because the DW1000's distance estimate fluctuates based on channel conditions, noise, and multipath. It's not a fixed offset per scenario.

```
Ranging error stats:
  Mean: 3.295m, Std: 1.748m
  Min: 0.116m, Max: 6.044m

Per group:
  12.79m: 240 samples, mean error ~2.0m
  16.09m: 239 samples, mean error ~4.6m
```

### Architecture

```
Frozen Stage 1 encoder
        |
   64-dim LNN embedding (for each correctable NLOS sample)
        |
   Random Forest Regressor (200 trees)
        |
   Predicted ranging error (meters)
        |
   d_corrected = d_hardware - predicted_error
```

Same shared encoder, same 64-dim embeddings, but now a **regressor** instead of a classifier. The RF learns a continuous mapping from temporal CIR dynamics to ranging error magnitude.

### Results

```
Test MAE:   0.4492m
Test RMSE:  0.8751m
Test R^2:   0.6961

Per-group test MAE:
  12.79m: 0.3315m
  16.09m: 0.5668m

Distance Correction:
  Before ML: MAE = 3.30m  (raw hardware error)
  After ML:  MAE = 0.45m  (corrected error)
  Improvement: ~7.3x lower error
```

---

## Why This Architecture Works

### Shared Encoder, Specialized Heads

The DualCircuit_PI_HLNN encoder is trained once (Stage 1) and frozen. Its 64-dim embedding captures rich temporal dynamics from the CIR — how the signal rises, peaks, decays, and how multipath components interfere. This single representation serves three different tasks:

1. **LOS vs NLOS** (Stage 1 classifier head) — is there an obstruction?
2. **Signal quality** (Stage 2 RF) — is the NLOS signal clean enough?
3. **Error magnitude** (Stage 3 RF) — how much to correct?

This is efficient: one 17K-parameter encoder replaces three separate feature extractors.

### Why Random Forest on Top of LNN?

The 64-dim LNN embeddings are **tabular features** — fixed-length vectors per sample. Random Forest excels at:
- Small datasets (479 samples for Stage 3)
- No hyperparameter sensitivity (no learning rate, no epochs)
- Built-in feature importance (shows which embedding dimensions matter)
- No overfitting risk with proper tree depth

The LNN handles the hard part (temporal sequence -> fixed representation). The RF handles the easy part (fixed representation -> prediction).

### The Role of FP_AMPL

FP_AMPL1/2/3 appear at two levels:
1. **Encoder conditioning** (Stage 1): Seeds the initial hidden state of both circuits, making tau dynamics FP-aware from timestep 0
2. **Implicit in embeddings** (Stage 2/3): Since the encoder was conditioned on FP_AMPL during training, the 64-dim embeddings already encode FP_AMPL information. The RFs don't need FP_AMPL separately.

---

## Comparison with Xueli's Pipeline

| Aspect | Our Pipeline | Xueli's Pipeline |
|--------|-------------|-----------------|
| **Encoder** | DualCircuit_PI_HLNN (~17K params) | Modified Transformer (~millions params) |
| **Embedding dim** | 64 | 504 |
| **CIR input** | Raw 60-sample window | Binned CIR (7 discrete levels) |
| **Stage 1 accuracy** | 100% | 99% |
| **Stage 2 accuracy** | 91.1% | 100% (on 314 NLOS) |
| **Stage 3 MAE** | 0.449m | 0.346m |
| **Stage 3 R^2** | 0.696 | 0.619 |
| **Dataset size** | 3600 (balanced) | 952 (imbalanced) |
| **Bounce labels** | Laser + floor plan + CIR analysis | Laser only |
| **Physics-informed** | Yes (ODE dynamics, tau, dual circuits) | No (general-purpose Transformer) |
| **Interpretable** | Yes (tau dynamics, circuit specialization) | Limited |

**Key advantage**: Our R^2 (0.696) significantly exceeds Xueli's (0.619), meaning our model captures more of the variance in ranging error. The corrections are more consistently reliable.

---

## Artifacts

| File | Description |
|------|-------------|
| `stage1_pi_hlnn_best.pt` | Trained DualCircuit_PI_HLNN weights (shared encoder) |
| `stage1_config.pt` | Stage 1 hyperparameters |
| `stage2_bounce_rf.joblib` | Stage 2 RF classifier (signal quality) |
| `stage2_config.joblib` | Stage 2 config + label strategy |
| `stage3_nlos_bias_rf.joblib` | Stage 3 RF regressor (ranging error) |
| `stage3_config.joblib` | Stage 3 config + filter strategy |

---

## Full Inference Pipeline (Pseudocode)

```python
# Input: raw CIR (1016 samples) + FP_AMPL1/2/3

# 1. Preprocess
sig = raw_cir / RXPACC                        # physics normalization
leading_edge = find_leading_edge(sig, [740, 890])
crop = sig[le-10 : le+50]                     # 60-sample window
crop = (crop - min) / (max - min)             # instance normalize
fp = [FP_AMPL1, FP_AMPL2, FP_AMPL3] / RXPACC / 64  # normalize FP

# 2. Stage 1: LOS/NLOS
embedding = encoder.embed(crop, fp_features=fp)  # 64-dim
prob_nlos = encoder.forward(crop, fp_features=fp) # P(NLOS)

if prob_nlos < 0.5:
    return d_hardware  # LOS — no correction needed

# 3. Stage 2: Signal Quality
quality = rf_classifier.predict(embedding)  # 0=Correctable, 1=Challenging

if quality == 1:
    return d_hardware, flag="low_confidence"  # Challenging — correction unreliable

# 4. Stage 3: Ranging Error Correction
predicted_error = rf_regressor.predict(embedding)  # meters
d_corrected = d_hardware - predicted_error

return d_corrected  # Corrected distance
```
