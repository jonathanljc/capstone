# DualCircuit Physics-Informed Hybrid Liquid Neural Network (PI-HLNN)

## Architecture Overview

The DualCircuit_PI_HLNN is a binary classifier (LOS vs NLOS) that processes raw Channel Impulse Response (CIR) sequences from UWB sensors. It uses two parallel conductance-based Liquid Time-Constant (LTC) neural circuits with cross-circuit communication.

**Input**: Raw CIR window of 60 timesteps x 1 feature (normalised amplitude) + FP_AMPL1/2/3 (first-path amplitudes)
**Output**: P(NLOS) in [0, 1]
**Total Parameters**: ~17,200
**Embedding Dimension**: 64 (used by downstream Stage 2 and Stage 3)

---

## High-Level Architecture Diagram

```
                        ┌─────────────────────────┐
                        │   Raw CIR Input          │
                        │   (batch, 60, 1)         │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │          SHARED INPUT            │
                    │   (same CIR fed to both cells)   │
                    └────────────────┬────────────────┘
                                     │
        ┌────────────────────────────┤
        │                            │
        │   ┌──────────────────┐     │
        │   │  FP_AMPL1/2/3    │     │
        │   │  (batch, 3)      │     │
        │   └────────┬─────────┘     │
        │            │               │
        │   ┌────────▼─────────┐     │
        │   │ fp_to_los_init   │     │
        │   │ Linear(3→32)     │     │
        │   │ → 0.1×tanh(·)   │     │
        │   │ = h_los_0        │     │
        │   └──────────────────┘     │
        │            │               │
        │   ┌────────▼─────────┐     │
        │   │ fp_to_nlos_init  │     │
        │   │ Linear(3→32)     │     │
        │   │ → 0.1×tanh(·)   │     │
        │   │ = h_nlos_0       │     │
        │   └──────────────────┘     │
        │            │               │
                  ┌──┴───────────────┴──────────────────┐
                  ▼                                      ▼
        ┌─────────────────┐                   ┌─────────────────┐
        │   PILiquidCell  │                   │   PILiquidCell  │
        │    (cell_los)   │◄─────────────────►│   (cell_nlos)   │
        │   32 neurons    │  Cross-Circuit    │   32 neurons    │
        │   init: h_los_0 │  Gated Projections│   init: h_nlos_0│
        └────────┬────────┘                   └────────┬────────┘
                 │          (at every timestep)         │
                 │                                      │
                 ▼                                      ▼
        ┌─────────────────┐                   ┌─────────────────┐
        │ Attention Pool  │                   │ Attention Pool  │
        │ (60 steps → 1)  │                   │ (60 steps → 1)  │
        │ Linear(32→1)    │                   │ Linear(32→1)    │
        └────────┬────────┘                   └────────┬────────┘
                 │                                      │
              (32-dim)                              (32-dim)
                 │                                      │
                 └──────────────┬───────────────────────┘
                                │
                           Concatenate
                            (64-dim)
                                │
                                ▼
                   ┌────────────────────────┐
                   │      Classifier        │
                   │  Linear(64 → 32)       │
                   │  SiLU activation       │
                   │  Dropout(0.2)          │
                   │  Linear(32 → 1)        │
                   │  Sigmoid               │
                   └────────────┬───────────┘
                                │
                                ▼
                         P(NLOS) ∈ [0, 1]
```

---

## Component 1: PILiquidCell (Conductance-Based LTC Neuron)

Each PILiquidCell is a group of 32 interconnected neurons that evolve according to a continuous-time ODE (Ordinary Differential Equation), inspired by biological neuron membrane dynamics.

### Biological Analogy

```
              OUTSIDE CELL MEMBRANE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        │              │              │
    ┌───┴───┐    ┌─────┴─────┐   ┌───┴────┐
    │ Leak  │    │ Recurrent │   │Sensory │
    │Channel│    │ Synapses  │   │Input   │
    │g_leak │    │ w, erev   │   │from CIR│
    └───┬───┘    └─────┬─────┘   └───┬────┘
        │              │              │
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              INSIDE CELL (voltage v = hidden state)
```

### The ODE Governing Each Neuron

```
Cm × dv/dt = -g_leak × (v - v_leak)           ← leak current (pulls toward rest)
             - Σ w_ij × gate_ij × (v - erev_ij) ← recurrent synaptic current
             + I_sensory                         ← input current from CIR signal
```

Solved via semi-implicit Euler with 6 sub-steps (ode_unfolds=6) per timestep.

### Three Current Types

#### 1. Leak Current (the baseline drain)
```
I_leak = g_leak × (v - v_leak)

g_leak: [32] — leak conductance per neuron (learned, softplus-constrained > 0)
v_leak: [32] — resting voltage per neuron (learned)
```
Always pulls the neuron voltage back toward v_leak. This is the "forgetting" mechanism.

#### 2. Recurrent Synaptic Current (neuron-to-neuron communication)
```
For each neuron pair (i, j) out of 32×32:
    gate_ij = sigmoid(sigma_ij × (v_j - mu_ij))     ← activation gate
    I_recurrent_ij = w_ij × gate_ij × (v_i - erev_ij) ← conductance current

Parameters (all 32×32 matrices):
    w:     synaptic weight (conductance strength, softplus > 0)
    mu:    activation center ("neuron j must reach this voltage to activate")
    sigma: activation sharpness (steepness of the sigmoid gate)
    erev:  reversal potential (voltage that neuron j pushes neuron i TOWARD)
           - If erev > v: excitatory (pushes voltage up)
           - If erev < v: inhibitory (pushes voltage down)
```

#### 3. Sensory Current (CIR signal input)
```
For each input feature × neuron (1×32):
    sensory_gate = sigmoid(sensory_sigma × (x_t - sensory_mu))
    I_sensory = sensory_w × sensory_gate × x_t

Parameters:
    sensory_w:     [1, 32] — input weight per neuron
    sensory_mu:    [1, 32] — input threshold ("what CIR amplitude is significant?")
    sensory_sigma: [1, 32] — input gate sharpness
```

When CIR amplitude is below sensory_mu (noise floor), the gate ≈ 0 and input is ignored.
When CIR amplitude exceeds sensory_mu (peak region), the gate ≈ 1 and input current flows in.

### Semi-Implicit Euler Solver (6 sub-steps)

The ODE is solved iteratively, not in closed form:

```
For each sub-step k = 1..6:
    dt_sub = dt / 6
    Cm_t = Cm / dt_sub

    recurrent_gate = sigmoid(sigma × (v - mu))        ← which synapses are active?
    w_gate = w × recurrent_gate                        ← effective conductance
    w_num = Σ(w_gate × erev)                           ← numerator contribution
    w_den = Σ(w_gate)                                  ← denominator contribution

    v = (Cm_t × v + g_leak × v_leak + w_num + I_sensory)
        ─────────────────────────────────────────────────
                    Cm_t + g_leak + w_den + ε

# After all 6 sub-steps: clamp hidden state for numerical stability
v = clamp(v, -1.0, 1.0)
```

### State Clamping

After the ODE solver completes all 6 sub-steps, the hidden state is clamped to [-1, 1]:
```python
v = torch.clamp(v, -1.0, 1.0)
```
This prevents numerical instability during training — without clamping, the ODE dynamics can occasionally diverge (especially early in training when parameters are random), causing NaN gradients.

### Emergent Time Constant (Tau)

After the ODE solve, the effective time constant is:

```
τ = Cm / (g_leak + Σ(w × gate(v)))
```

- **Large tau** → neuron changes slowly → preserves memory of past timesteps
- **Small tau** → neuron responds quickly → reacts to current input

Tau is NOT a fixed hyperparameter. It adapts at every timestep based on:
- The current hidden state v (via gate(v))
- The input signal strength (via sensory current affecting v)

This is why it's called "Liquid" — the time constants flow and adapt.

### PILiquidCell Parameter Count (per cell)

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| gleak | [32] | 32 | Leak conductance |
| vleak | [32] | 32 | Resting potential |
| cm | [32] | 32 | Membrane capacitance |
| w | [32, 32] | 1,024 | Recurrent synapse weight |
| erev | [32, 32] | 1,024 | Reversal potential |
| mu | [32, 32] | 1,024 | Recurrent activation center |
| sigma | [32, 32] | 1,024 | Recurrent activation sharpness |
| sensory_w | [1, 32] | 32 | Input weight |
| sensory_mu | [1, 32] | 32 | Input activation center |
| sensory_sigma | [1, 32] | 32 | Input activation sharpness |
| **Total per cell** | | **4,288** | |
| **Total (2 cells)** | | **8,576** | |

---

## Component 2: Cross-Circuit Gated Communication

At every timestep, the two circuits exchange information through learned gated projections:

```
Timestep t:
┌──────────┐                                    ┌──────────┐
│ h_los(t) │──── P_los2nlos ────► proj ────┐    │h_nlos(t) │
│          │                                │    │          │
│          │    ┌── proj ◄── P_nlos2los ────┼────│          │
└──────────┘    │                           │    └──────────┘
                ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │ gate_los =        │       │ gate_nlos =       │
    │ σ(W×[h_los|proj]) │       │ σ(W×[h_nlos|proj])│
    └─────────┬─────────┘       └─────────┬─────────┘
              │                           │
              ▼                           ▼
    h_los_in = h_los              h_nlos_in = h_nlos
             + gate_los × proj             + gate_nlos × proj
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │ cell_los ODE    │         │ cell_nlos ODE   │
    │ solve step      │         │ solve step      │
    └─────────────────┘         └─────────────────┘
```

### Cross-Circuit Parameters

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| P_nlos2los | [32, 32] | 1,024 | Project NLOS state → LOS space |
| P_los2nlos | [32, 32] | 1,024 | Project LOS state → NLOS space |
| gate_los (W + b) | [64, 32] + [32] | 2,080 | How much LOS circuit accepts from NLOS |
| gate_nlos (W + b) | [64, 32] + [32] | 2,080 | How much NLOS circuit accepts from LOS |
| **Total** | | **6,208** | |

### Why Cross-Circuit Communication Matters

- The NLOS circuit can **slow down** the LOS circuit's tau when it detects multipath complexity
- The LOS circuit can **sharpen** the NLOS circuit when it sees a clean single peak
- The sigmoid gate learns WHEN to listen vs. WHEN to ignore the other circuit
- This implements an implicit "confidence exchange" between the two interpretations of the signal

---

## Component 2b: FP_AMPL Conditioning (Initial Hidden State Seeding)

Instead of zero-initialising the hidden state of each circuit, the DW1000's first-path amplitude measurements (FP_AMPL1, FP_AMPL2, FP_AMPL3) seed the initial state:

```
FP_AMPL1/2/3 (normalised by RXPACC/64)
        │
        ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ fp_to_los_init           │     │ fp_to_nlos_init          │
│ Linear(3 → 32, no bias) │     │ Linear(3 → 32, no bias) │
│ → tanh(·)               │     │ → tanh(·)               │
│ → × 0.1                 │     │ → × 0.1                 │
│ = h_los_0               │     │ = h_nlos_0              │
└──────────────────────────┘     └──────────────────────────┘
```

```python
h_los_0  = 0.1 * tanh(fp_to_los_init(fp_features))    # (batch, 32)
h_nlos_0 = 0.1 * tanh(fp_to_nlos_init(fp_features))   # (batch, 32)
```

### Why 0.1 Scaling?

The 0.1 factor is critical — it provides a **gentle nudge** rather than a strong bias:
- Too large (e.g. 1.0): the model shortcuts from FP amplitudes alone, ignoring CIR temporal dynamics
- Too small (e.g. 0.0): no hardware prior, circuits start from zero (wastes the FP information)
- 0.1: gives a hint about channel conditions but forces the model to develop its understanding from the actual CIR waveform

### Physics Connection to Tau

Since `τ = Cm / (g_leak + Σ(w × gate(v)))` and `v_0 = h_los_0` or `h_nlos_0` comes from FP_AMPL, the time constant is immediately influenced by the hardware's first-path measurement from timestep 0:
- **Strong FP** (likely LOS): starts the circuit in one dynamical regime
- **Weak FP** (likely NLOS): starts it in another regime
- Tau then evolves as the CIR waveform is processed timestep by timestep

### FP_AMPL Conditioning Parameters

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| fp_to_los_init (W) | [3, 32] | 96 | FP_AMPL → LOS initial state |
| fp_to_nlos_init (W) | [3, 32] | 96 | FP_AMPL → NLOS initial state |
| **Total** | | **192** | |

---

## Component 3: Attention Pooling

After 60 timesteps, each circuit has produced a sequence of hidden states:

```
cell_los  → los_states:  (batch, 60, 32)
cell_nlos → nlos_states: (batch, 60, 32)
```

Instead of using only the last timestep (which discards temporal information), learned attention selects the most informative timesteps:

```
Attention for LOS circuit:
    score_t = Linear(32 → 1)(los_states[:, t, :])    for t = 0..59
    weights = softmax(scores)                          ← (batch, 60)
    h_los_pooled = Σ(los_states × weights)             ← (batch, 32)

Same for NLOS circuit → h_nlos_pooled: (batch, 32)
```

The model typically learns to attend to the **leading edge and peak region** (timesteps ~9-15 in the 60-step window) where LOS/NLOS differences are most pronounced.

### Attention Parameters

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| los_attn (W + b) | [32, 1] + [1] | 33 | LOS temporal attention |
| nlos_attn (W + b) | [32, 1] + [1] | 33 | NLOS temporal attention |
| **Total** | | **66** | |

---

## Component 4: Fusion and Classifier

```
h_los_pooled (32) ─┐
                    ├── concatenate ── (64-dim fused embedding)
h_nlos_pooled (32)─┘
                                          │
                                          ▼
                                 Linear(64 → 32)
                                          │
                                       SiLU()
                                          │
                                    Dropout(0.2)
                                          │
                                 Linear(32 → 1)
                                          │
                                      Sigmoid()
                                          │
                                    P(NLOS) ∈ [0,1]
```

The 64-dim embedding is also used by Stage 2 (bounce classifier) and Stage 3 (ranging error regressor) via the `model.embed()` method.

### Classifier Parameters

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| Linear 1 (W + b) | [64, 32] + [32] | 2,080 | Fused embedding → hidden |
| Linear 2 (W + b) | [32, 1] + [1] | 33 | Hidden → output |
| **Total** | | **2,113** | |

---

## Complete Parameter Summary

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| cell_los (PILiquidCell) | 4,288 | 24.9% |
| cell_nlos (PILiquidCell) | 4,288 | 24.9% |
| FP_AMPL conditioning (2 × Linear(3→32)) | 192 | 1.1% |
| Cross-circuit projections | 2,048 | 11.9% |
| Cross-circuit gates | 4,160 | 24.2% |
| Attention pooling | 66 | 0.4% |
| Classifier | 2,113 | 12.3% |
| **Total** | **~17,155** | **100%** |

---

## Data Flow: Processing One CIR Sample

```
Step 1: PREPROCESSING (before model)
   Raw CIR (1016 samples)
   → Divide by RXPACC (physics normalisation)
   → ROI alignment (find leading edge in [740, 890] window)
   → Crop 60 samples (10 pre-peak + 50 post-peak)
   → Min-max normalise to [0, 1]
   → Shape: (60, 1)

Step 2: FP_AMPL CONDITIONING (initial state seeding)
   fp = [FP_AMPL1, FP_AMPL2, FP_AMPL3] / RXPACC / 64   # normalise
   h_los_0  = 0.1 × tanh(Linear(3→32)(fp))               # gentle nudge for LOS circuit
   h_nlos_0 = 0.1 × tanh(Linear(3→32)(fp))               # gentle nudge for NLOS circuit

Step 3: DUAL-CIRCUIT PROCESSING (60 timesteps)
   Initialise: h_los = h_los_0, h_nlos = h_nlos_0
   For t = 0 to 59:
     a. Read CIR amplitude x_t
     b. Cross-circuit projection: project h_nlos → LOS space, h_los → NLOS space
     c. Gating: sigmoid gates decide how much cross-info to accept
     d. Mix: h_los += gate × projected_nlos (and vice versa)
     e. ODE solve: both cells run 6 Euler sub-steps with x_t as sensory input
     f. Clamp: v = clamp(v, -1.0, 1.0) for numerical stability
     g. Store hidden states and tau values

Step 4: ATTENTION POOLING
   LOS circuit:  60 hidden states → weighted sum → 32-dim
   NLOS circuit: 60 hidden states → weighted sum → 32-dim

Step 5: FUSION
   Concatenate: [h_los_pooled | h_nlos_pooled] → 64-dim embedding

Step 6: CLASSIFICATION
   64-dim → Linear → SiLU → Dropout → Linear → Sigmoid → P(NLOS)
```

---

## What Makes This Architecture Unique

### "Physics-Informed"
1. **RXPACC normalisation**: CIR divided by preamble accumulation count (hardware-specific calibration)
2. **ODE-based dynamics**: neurons evolve via continuous differential equations, not discrete gates
3. **Conductance model**: synapses use reversal potentials (erev), matching biological neuron physics
4. **Emergent tau**: time constants arise from signal-state interaction, not from fixed hyperparameters

### "Hybrid"
- Physics priors baked into preprocessing (RXPACC, ROI alignment) and the ODE structure
- All internal parameters (conductances, reversal potentials, thresholds, gates) are learned from data
- No hand-crafted features — the model learns directly from raw CIR waveforms

### "Liquid"
- Time constants (tau) are not fixed — they adapt continuously based on both input and internal state
- The model naturally speeds up when encountering sharp CIR transitions (peak onset)
- The model naturally slows down in noise regions (preserving prior information)

### "Dual-Circuit"
- Two parallel processing pathways that specialise through training
- Cross-circuit communication allows "consulting" the other interpretation
- Attention pooling independently determines which timesteps matter for each circuit
- Final decision is based on evidence from both pathways

---

## Training Configuration

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| hidden_size | 32 per circuit | Circuit capacity (64 total embedding) |
| ode_unfolds | 6 | ODE solver sub-steps per timestep |
| batch_size | 64 | Training batch size |
| max_epochs | 40 | Maximum training epochs |
| patience | 10 | Early stopping patience (restore best model) |
| learning_rate | 1e-3 | AdamW optimiser |
| weight_decay | 1e-4 | L2 regularisation |
| warmup_epochs | 3 | Linear LR warmup |
| LR schedule | Cosine annealing | After warmup, cosine decay to 1% of peak LR |
| grad_clip | 1.0 | Gradient norm clipping |
| dropout | 0.2 | In classifier head |
| loss | BCE (Binary Cross-Entropy) | No tau constraint in loss |
| seed | 42 | Fixed for reproducibility |
