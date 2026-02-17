# Multi-Stage UWB NLOS Mitigation Pipeline — Detailed Architecture

## Pipeline Overview

```
Raw CIR (1016 samples)
    │
    ▼
┌──────────────────────────────────┐
│  Stage 1: PI-HLNN                │
│  Task: LOS vs NLOS classification│
│  Model: Liquid Neural Network    │
│  Output: binary label + 48-dim   │
│          hidden state embeddings │
└──────────┬───────────────────────┘
           │
           │  Freeze encoder weights
           │  Extract 48-dim embeddings
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────────┐ (only NLOS samples pass through)
│            │
│  Stage 2: Random Forest Classifier          │
│  Task: Single-bounce vs Multi-bounce        │
│  Input: 48-dim LNN embeddings               │
│  Output: bounce type label                  │
└────────────┬────────────────────────────────┘
             │
             │ (only single-bounce samples pass through)
             ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Random Forest Regressor           │
│  Task: Predict NLOS ranging bias (meters)   │
│  Input: 48-dim LNN embeddings               │
│  Output: predicted bias → d_corrected       │
└─────────────────────────────────────────────┘
```

**Correction formula**: `d_corrected = d_UWB - predicted_bias ≈ d_direct`

---

## Shared Preprocessing

All three stages share the same CIR preprocessing pipeline before anything else happens.

### Raw CIR Signal
- **Source**: DW1000 UWB transceiver Channel Impulse Response
- **Raw length**: 1016 samples per measurement
- **Dataset**: 3600 total samples (1800 LOS + 1800 NLOS)
- **Split**: 70% train (2520) / 15% val (540) / 15% test (540), stratified by label

### RXPACC Normalization
```
sig = sig / RXPACC
```
- RXPACC = Receiver Preamble Accumulation Count from the DW1000
- This is a **physics-informed** step: normalizes for preamble accumulation differences across measurements

### ROI (Region of Interest) Alignment
The goal is to find the **leading edge** — the point where the first-path signal arrives.

1. **Search for peak** in region [740, 890] of the CIR
2. **Estimate noise floor**: mean + 3×std of samples before index 740
3. **Set threshold**: max(noise_floor, 5% of peak amplitude)
4. **Backtrack from peak** to find where signal drops below threshold → that's the leading edge
5. **Crop window**: 10 samples before leading edge, 50 after → **60-sample window**
6. **Min-max normalize** the window to [0, 1]

```
Final input shape per sample: (60, 1)
```

---

## Stage 1: PI-HLNN (Physics-Informed Hybrid Liquid Neural Network)

### Purpose
Binary classification: **LOS (0) vs NLOS (1)** from the 60-sample CIR window.

### Why Liquid Neural Networks?
Standard RNNs use fixed time constants. Liquid Neural Networks (LNNs) model neurons as **leaky integrators governed by ODEs**, where the time constant tau adapts dynamically based on input. This is well-suited for CIR analysis because:
- CIR signals have varying temporal characteristics depending on the propagation environment
- A neuron that can slow down (large tau) or speed up (small tau) its response based on what it sees can better capture multipath dynamics

### Architecture

```
PI_HLNN (7,233 parameters)
├── PILiquidCell (recurrent cell, processes 1 timestep at a time)
│   ├── synapse:  Linear(49 → 48)     — maps [x_t, h_prev] to synaptic input
│   ├── tau_net:  Linear(49 → 32) → Tanh → Linear(32 → 48)  — predicts time constants
│   └── A:        Parameter(48)        — learnable decay rates, initialized to -0.5
│
└── Classifier (applied to mean-pooled hidden state after all 60 timesteps)
    ├── Linear(48 → 32)
    ├── SiLU activation
    ├── Dropout(0.4)
    ├── Linear(32 → 1)
    └── Sigmoid
```

### ODE Dynamics (the core of the LNN)

At each timestep t, the cell computes:

```
combined = concat(x_t, h_prev)           # shape: (batch, 49)

tau_t    = dt + softplus(tau_net(combined))   # shape: (batch, 48)
S_t      = tanh(synapse(combined))            # shape: (batch, 48)

h_new    = (h_prev + dt * S_t * A) / (1 + dt / tau_t)
```

**What each component does**:

| Component | Formula | Role |
|-----------|---------|------|
| `tau_t` | `dt + softplus(...)` | **Time constant** — how quickly each neuron responds. Minimum value = dt (=1.0), ensuring stability. Larger tau = slower response (memory persists). Smaller tau = faster response (reacts to new input). |
| `S_t` | `tanh(synapse([x_t, h_prev]))` | **Synaptic input** — the "force" driving the neuron. Combines current CIR sample with previous state. |
| `A` | Learnable parameter | **Decay rate** — initialized to -0.5, controls how the driving force scales. Negative values create stable dynamics. |
| `h_new` | `(h + dt*S*A) / (1 + dt/tau)` | **Euler discretization** of the ODE: `tau * dh/dt = -h + S*A`. This is the leaky integrator equation. |

**Why softplus for tau?**
- Previous version used `1 + 5 * sigmoid(...)` → tau range [1, 6] (artificially bounded)
- Current version uses `dt + softplus(...)` → tau range [1, ∞) (naturally bounded below, unconstrained above)
- Softplus is smooth and differentiable everywhere, better for gradient flow

### Pooling: From Sequence to Fixed Vector

After processing all 60 timesteps, we have 60 hidden states. These are aggregated into a single vector:

```
h_pooled = (h_1 + h_2 + ... + h_60) / 60    # shape: (batch, 48)
```

This **mean-pooled hidden state** serves two purposes:
1. **Stage 1**: Fed into the classifier head for LOS/NLOS prediction
2. **Stages 2 & 3**: Used as the 48-dim embedding (feature vector) for downstream Random Forests

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Max epochs | 40 |
| Warmup | 3 epochs (linear LR ramp) |
| Early stopping | patience = 10 epochs |
| Gradient clipping | max norm = 0.5 |
| Loss function | Binary Cross-Entropy (BCE) |

### Output
- **Prediction**: scalar in [0, 1], thresholded at 0.5 → LOS (0) or NLOS (1)
- **Tau mean**: 48-dim vector of average time constants (diagnostic, not used in loss)
- **Validation accuracy**: ~94.6%

### Freezing for Downstream Use

After Stage 1 training completes:
```python
model_s1.eval()
for param in model_s1.parameters():
    param.requires_grad = False
```
The encoder weights are **permanently frozen**. Stages 2 and 3 only read embeddings — they never modify the encoder.

---

## Embedding Extraction (Shared by Stages 2 & 3)

This is how the frozen Stage 1 encoder becomes a feature extractor:

```python
def extract_lnn_embeddings(model, data_df, batch_size=256):
    X, _ = preprocess_stage1(data_df)        # (N, 60, 1)
    X_tensor = torch.tensor(X).to(device)

    all_embeddings = []
    with torch.no_grad():
        for batch in batches(X_tensor, batch_size):
            _, h_hist, _, _ = model(batch, return_dynamics=True)
            # h_hist shape: (batch, 60, 48) — all 60 hidden states
            emb = h_hist.mean(dim=1)   # (batch, 48) — mean pool over time
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)   # (N, 48)
```

**What the 48 dimensions represent**:
Each dimension (`LNN_h0` through `LNN_h47`) is one neuron's average activation across the 60 timesteps. The encoder learned these representations during LOS/NLOS classification — they capture signal characteristics like:
- Multipath structure (multiple reflections show up as temporal patterns)
- Signal decay rate (how quickly energy drops after the peak)
- Delay spread (how wide the impulse response is)
- Peak sharpness (concentrated vs diffuse energy)

These are richer than the 3 hand-crafted features because the network discovers whatever patterns are most discriminative, rather than relying on human-designed metrics.

---

## Stage 2: Random Forest Bounce Classifier

### Purpose
Classify NLOS signals into **single-bounce (0)** vs **multi-bounce (1)**.

This is the novel contribution — identifying single-bounce NLOS signals that can be corrected.

### Why Only NLOS Samples?
LOS signals don't have bounce reflections. Only samples classified as NLOS by Stage 1 enter Stage 2.

### Auto-Labeling (Ground Truth for Bounce Type)
There are no human-annotated bounce labels. Instead, labels are derived automatically from CIR peak analysis:

```python
# Count peaks in the ROI around the leading edge
peaks = find_peaks(roi_normalized, prominence=0.20, distance=5)

# Label rule:
#   Num_Peaks <= 2  →  single-bounce (0)
#   Num_Peaks > 2   →  multi-bounce  (1)
```

**Rationale**: A single-bounce reflection produces 1-2 dominant peaks (direct + one reflection). Multi-bounce produces more peaks from multiple reflection paths.

### Input Features
```
48-dim LNN embeddings from frozen Stage 1 PI-HLNN encoder
Shape: (N_nlos, 48)
```

These replace the previously used 3 hand-crafted features (Kurtosis, RMS_Delay_Spread, Power_Ratio).

### Model

```
RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=None,          # trees grow to full depth
    min_samples_split=5,     # minimum samples to split a node
    min_samples_leaf=2,      # minimum samples in a leaf
    class_weight='balanced', # upweight minority class
    n_jobs=-1                # use all CPU cores
)
```

**Why Random Forest?**
- No scaling needed (trees are scale-invariant)
- Handles the 48-dim embedding space well
- Fast training, no hyperparameter-sensitive training loop
- `class_weight='balanced'` handles the slight class imbalance (~50/50 in this case)

### Data Flow

```
Train NLOS (1260 samples)
    │
    ├── extract_features_from_df() → Num_Peaks → auto-labels (used for y)
    │
    └── extract_lnn_embeddings()   → (1260, 48) embeddings (used for X)

RF.fit(X_train, y_train)
```

### Performance
- **Train accuracy**: ~97.6%
- **Validation accuracy**: ~84.4%
- **Test accuracy**: ~89.2%

---

## Stage 3: Random Forest Bias Regressor

### Purpose
Predict the **NLOS ranging bias** (in meters) for single-bounce NLOS signals, enabling distance correction.

### Why Only Single-Bounce?
Multi-bounce signals have complex, unpredictable reflection paths. Single-bounce signals follow a known geometric model: the signal reflects off one surface, traveling a longer path than the direct line-of-sight distance. This extra path length is the bias.

### Ground Truth Bias (Physically Measured)

These values were measured in the physical experiment setup:

| Setup | d_direct | d_bounce (d₁+d₂) | Bias = d_bounce - d_direct |
|-------|----------|-------------------|---------------------------|
| 7.79m | 7.79m | 12.79m | **5.00m** |
| 10.77m | 10.77m | 16.09m | **5.32m** |
| 14m | 14.00m | 16.80m | **2.80m** |

Each sample's bias target is looked up by its distance group (from the source filename).

### Input Features
```
48-dim LNN embeddings from frozen Stage 1 PI-HLNN encoder
Shape: (N_single_bounce, 48)
```

**Key efficiency**: Stage 3 doesn't re-extract embeddings. It **indexes into Stage 2's embedding array** for the single-bounce subset:
```python
X_train_s3 = X_train_s2[single_bounce_indices]  # subset of Stage 2 embeddings
```

### Model

```
RandomForestRegressor(
    n_estimators=200,        # 200 decision trees
    max_depth=None,          # trees grow to full depth
    min_samples_split=5,     # minimum samples to split a node
    min_samples_leaf=2,      # minimum samples in a leaf
    n_jobs=-1                # use all CPU cores
)
```

### Data Flow

```
NLOS embeddings from Stage 2: (1260, 48)
    │
    ├── Filter: Num_Peaks <= 2 AND has known ground truth bias
    │
    └── Single-bounce subset: (~586, 48) with bias targets [5.00, 5.32, 2.80]

RF.fit(X_train, y_train)
```

### Performance
- **Train MAE**: ~0.01m
- **Validation MAE**: ~0.018m
- **Test MAE**: ~0.05m

### Distance Correction (Final Output)

```
d_corrected = d_UWB - predicted_bias

Example:
  d_UWB = 12.79m (biased measurement)
  predicted_bias = 4.98m
  d_corrected = 12.79 - 4.98 = 7.81m
  d_direct (truth) = 7.79m  ← close!
```

---

## End-to-End Inference Flow

At test time, the full pipeline runs in sequence:

```
Input: 540 test CIR signals
         │
         ▼
    Stage 1: PI-HLNN classifies LOS/NLOS
         │
         ├── LOS (289 predicted) → done, no correction needed
         │
         └── NLOS (251 predicted) → continue
                  │
                  ▼
             Stage 2: RF classifies bounce type
                  │
                  ├── Multi-bounce (138) → done, cannot reliably correct
                  │
                  └── Single-bounce (113) → continue
                           │
                           ▼
                      Stage 3: RF predicts bias
                           │
                           ▼
                      d_corrected = d_UWB - bias
```

### Pipeline Funnel (Test Set)
| Stage | Samples | Action |
|-------|---------|--------|
| Input | 540 | All test CIR signals |
| After Stage 1 | 251 | Classified as NLOS |
| After Stage 2 | 113 | Classified as single-bounce |
| After Stage 3 | 113 | Bias predicted, distance corrected |

---

## Summary Table

| Aspect | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| **Task** | LOS/NLOS classification | Bounce type classification | Bias prediction |
| **Model** | PI-HLNN (LNN) | Random Forest Classifier | Random Forest Regressor |
| **Parameters** | 7,233 | 200 trees | 200 trees |
| **Input** | Raw 60-sample CIR window | 48-dim LNN embeddings | 48-dim LNN embeddings |
| **Input shape** | (N, 60, 1) | (N_nlos, 48) | (N_single, 48) |
| **Output** | LOS (0) / NLOS (1) | Single (0) / Multi (1) | Bias in meters |
| **Loss/Metric** | BCE / Accuracy | — / Accuracy | — / MAE |
| **Training** | AdamW, 40 epochs | .fit() | .fit() |
| **Key property** | Learns ODE dynamics | Scale-invariant | Scale-invariant |

### Saved Artifacts
| File | Contents |
|------|----------|
| `prod_stage1_pi_hlnn.pt` | Frozen PI-HLNN weights (PyTorch) |
| `prod_stage2_bounce_rf.joblib` | Trained RF bounce classifier (sklearn) |
| `prod_stage3_bias_rf.joblib` | Trained RF bias regressor (sklearn) |
| `prod_pipeline_config.pt` | All configs, ground truth, peak settings |
