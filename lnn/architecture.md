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
│  Output: binary label + 64-dim   │
│          LNN embeddings          │
└──────────┬───────────────────────┘
           │
           │  Freeze encoder weights
           │  Extract 64-dim embeddings
           │  (mean-pooled hidden states)
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────────┐ (only NLOS samples pass through)
│            │
│  Stage 2: Random Forest Classifier      │
│  Task: Single-bounce vs Multi-bounce    │
│  Input: 64-dim LNN embeddings           │
│  Output: bounce type label              │
└────────────┬────────────────────────────┘
             │
             │ (only single-bounce samples pass through)
             ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Random Forest Regressor           │
│  Task: Predict NLOS ranging bias (meters)   │
│  Input: 64-dim LNN embeddings               │
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
PI_HLNN (~18,946 params — conductance-based LTC, Hasani et al.)
├── PILiquidCell (recurrent cell, processes 1 timestep via semi-implicit Euler)
│   ├── Neuron intrinsic parameters (per neuron):
│   │   ├── gleak:   Parameter(64)        — leak conductance (how fast neuron forgets)
│   │   ├── vleak:   Parameter(64)        — resting potential (equilibrium voltage)
│   │   └── cm:      Parameter(64)        — membrane capacitance (neuron sluggishness)
│   │
│   ├── Recurrent synapses (conductance-based, 64×64):
│   │   ├── w:       Parameter(64, 64)    — synaptic conductance strength
│   │   ├── erev:    Parameter(64, 64)    — reversal potential (excitatory/inhibitory)
│   │   ├── mu:      Parameter(64, 64)    — activation threshold per synapse
│   │   └── sigma:   Parameter(64, 64)    — gate sensitivity (sharpness)
│   │
│   └── Sensory synapses (gated additive input, 1×64):
│       ├── sensory_w:     Parameter(1, 64) — input conductance per neuron
│       ├── sensory_mu:    Parameter(1, 64) — CIR amplitude threshold
│       └── sensory_sigma: Parameter(1, 64) — input gate sharpness
│
├── Attention Pooling (learns which timesteps matter)
│   └── attn:         Linear(64 → 1)      — temporal importance scoring
│
└── Classifier (applied to attention-pooled hidden state after all 60 timesteps)
    ├── Linear(64 → 32)
    ├── SiLU activation
    ├── Dropout(0.4)
    ├── Linear(32 → 1)
    └── Sigmoid
```

### Parameter Count (~18,946 at h=64)

| Component | Formula | Count |
|-----------|---------|-------|
| Recurrent (w, erev, mu, sigma) | 4 × 64² | 16,384 |
| Sensory (w, mu, sigma) | 3 × 1 × 64 | 192 |
| Neuron (gleak, vleak, cm) | 3 × 64 | 192 |
| Attention | 64 + 1 | 65 |
| Classifier | (64×32 + 32) + (32×1 + 1) | 2,113 |
| **Total** | | **~18,946** |

**Comparable to LSTM (19,265)** — but with physics-interpretable ODE dynamics.

### Parameter Initialization (Tutorial-Style)

Following the conductance-based LTC tutorial (KPEKEP/LTCtutorial):

| Parameter | Init Range | Rationale |
|-----------|-----------|-----------|
| `gleak` | [0.001, 1.0] | Positive leak conductance |
| `vleak` | [-0.2, 0.2] | Small resting potential |
| `cm` | [0.4, 0.6] | Moderate capacitance |
| `w` | [0.001, 1.0] | Positive synaptic weights |
| `erev` | [-0.2, 0.2] | Mix of excitatory/inhibitory |
| `mu` | **[0.3, 0.8]** | Centered in active voltage range |
| `sigma` | **[3, 8]** | Sharp sensitivity for clear gating |
| `sensory_w` | [0.001, 1.0] | Positive input conductance |
| `sensory_mu` | [0.3, 0.8] | Centered in input range [0, 1] |
| `sensory_sigma` | [3, 8] | Sharp sensitivity for input gates |

### Positivity Enforcement: softplus on Conductances Only

Only parameters that represent **conductances** (must be positive) use `F.softplus(x) = log(1 + exp(x))`:
- **softplus applied**: `gleak`, `cm`, `w`, `sensory_w`
- **NOT applied**: `sigma`, `sensory_sigma` (gate sensitivity — allowed to use raw values)

This matches the tutorial's pattern: sigma controls gate sharpness and is initialized in a range [3, 8] where the raw value is already positive and meaningful. Applying softplus to sigma would compress the effective range and reduce gradient flow.

### ODE Dynamics (the core of the LNN)

The ODE per neuron i (Hodgkin-Huxley inspired):

```
cm_i * dV_i/dt = -gleak_i * (V_i - vleak_i)                                          ← leak current
                 + Σ_j [ w_ji * σ(sigma_ji * (V_j - mu_ji)) * (erev_ji - V_i) ]       ← recurrent synapses
                 + Σ_k [ sensory_w_ki * σ(sensory_sigma_ki * (I_k - sensory_mu_ki)) * I_k ]  ← sensory input
```

Solved with **semi-implicit Euler** (unconditionally stable, 6 sub-steps per timestep):

```
# Sensory current (computed once per timestep — independent of v):
sensory_gate    = sigmoid(sensory_sigma * (x_t - sensory_mu))     # (batch, input, hidden)
sensory_current = Σ_k (softplus(sensory_w) * sensory_gate * x_t)  # gated additive input

# Per ODE sub-step:
recurrent_gate  = sigmoid(sigma * (v_pre - mu))                   # (batch, h_pre, h_post)
w_gate          = softplus(w) * recurrent_gate

v_new = (cm_t * v + softplus(gleak) * vleak + Σ w_gate * erev + sensory_current)
      / (cm_t + softplus(gleak) + Σ w_gate + ε)

# After all timesteps:
attn_w   = softmax(attn(h_all), dim=time)                        # (batch, 60, 1)
h_pooled = sum(h_all * attn_w, dim=time)                          # (batch, 64)

# Emergent time constant (for diagnostics):
tau_eff = cm / (gleak + Σ w_gate)
```

**Key design choice**: The sensory current uses a **gated additive** formulation — it only contributes to the numerator, not the denominator. This keeps sensory input as a direct driving force on the neuron voltage without dampening responsiveness. The recurrent synapses use full conductance-based formulation (with reversal potentials in both numerator and denominator), which provides stability for the recurrent dynamics while allowing the input signal to flow freely.

**What each component does (white-box)**:

| Component | Role |
|-----------|------|
| `gleak` | **Leak conductance** — how quickly neuron i forgets (high = short memory) |
| `vleak` | **Resting potential** — value neuron relaxes to when unstimulated |
| `cm` | **Membrane capacitance** — neuron sluggishness (high = slow, low = fast) |
| `w[j,i]` | **Synaptic conductance** — strength of connection from neuron j to i |
| `erev[j,i]` | **Reversal potential** — determines excitatory (erev > V) or inhibitory (erev < V) |
| `mu[j,i]` | **Activation threshold** — neuron j must exceed mu for synapse to fire |
| `sigma[j,i]` | **Gate sensitivity** — how sharply the synapse turns on/off |
| `sensory_w` | **Input conductance** — how much CIR input drives each neuron |
| `sensory_mu` | **Input threshold** — what CIR amplitude activates this synapse |
| `attn_w` | **Attention pooling** — learned temporal importance weights |
| `tau_eff` | **Emergent time constant** — NOT explicitly set; emerges from cm, gleak, and synaptic activity |

**Why semi-implicit Euler?**
- LTC ODEs are **stiff** — standard explicit Euler would require very small steps
- Semi-implicit Euler is unconditionally stable: no gradient explosion from ODE dynamics
- 6 sub-steps per timestep balances accuracy vs computation cost

### Pooling: From Sequence to Fixed Vector

After processing all 60 timesteps, we have 60 hidden states. These are aggregated into a single vector:

```
attn_weights = softmax(attn(h_all))          # learned temporal importance
h_pooled = sum(h_all * attn_weights)         # shape: (batch, 64)
```

This **attention-pooled hidden state** serves two purposes:
1. **Stage 1**: Fed into the classifier head for LOS/NLOS prediction
2. **Stages 2 & 3**: Mean-pooled hidden states form 64-dim LNN embeddings for downstream Random Forests

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
- **Tau mean**: 64-dim vector of average time constants (used for diagnostics only)
- **Validation accuracy**: ~94.8%

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
            embedding = h_hist.mean(dim=1)   # (batch, 64) mean pooling
            all_embeddings.append(embedding.cpu().numpy())

    return np.vstack(all_embeddings)   # (N, 64)
```

### 64-dim Mean-Pooled LNN Embeddings

The embedding is the **mean of all hidden states across the 60 timesteps**:

```python
embedding = h_hist.mean(dim=1)  # (batch, 64)
```

| Dimensions | Source | What it captures |
|-----------|--------|-----------------|
| `LNN_0` – `LNN_63` | Mean-pooled hidden states | Signal structure, multipath patterns, temporal dynamics |

**Why mean pooling works well**:
- Each hidden state at timestep t contains the neuron voltages after processing the CIR up to that point
- Mean pooling aggregates information from all timesteps, capturing the full temporal evolution
- The 64-dim embeddings contain rich representations learned by the conductance-based ODE dynamics
- The effective time constant τ = cm/(gleak + Σ w·gate) is implicitly encoded in the hidden state evolution

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
64-dim LNN embeddings from frozen Stage 1 PI-HLNN encoder
Shape: (N_nlos, 64)
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
- Handles the 64-dim embedding space well
- Fast training, no hyperparameter-sensitive training loop
- `class_weight='balanced'` handles the slight class imbalance (~50/50 in this case)

### Data Flow

```
Train NLOS (1260 samples)
    │
    ├── extract_features_from_df() → Num_Peaks → auto-labels (used for y)
    │
    └── extract_lnn_embeddings()   → (1260, 64) embeddings (used for X)

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
64-dim LNN embeddings from frozen Stage 1 PI-HLNN encoder
Shape: (N_single_bounce, 64)
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
NLOS embeddings from Stage 2: (1260, 64)
    │
    ├── Filter: Num_Peaks <= 2 AND has known ground truth bias
    │
    └── Single-bounce subset: (~586, 64) with bias targets [5.00, 5.32, 2.80]

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
| **Model** | PI-HLNN (conductance-based LTC) | Random Forest Classifier | Random Forest Regressor |
| **Parameters** | ~18,946 (h=64) | 200 trees | 200 trees |
| **Input** | Raw 60-sample CIR window | 64-dim LNN embeddings | 64-dim LNN embeddings |
| **Input shape** | (N, 60, 1) | (N_nlos, 64) | (N_single, 64) |
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
