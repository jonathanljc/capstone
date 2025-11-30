# Architecture Diagram Refinements
## Context-Guided Multi-Scale LNN for UWB Localization

**Purpose:** Professional diagrams for capstone presentation and thesis documentation

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         UWB LOCALIZATION SYSTEM                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
        ┌───────────▼────────┐ ┌───▼────────┐ ┌───▼────────────┐
        │  Data Collection   │ │ Preprocessing│ │  LNN Model     │
        │  (DW1000 UWB)     │ │  & Features  │ │  (Classification)│
        └───────────┬────────┘ └───┬────────┘ └───┬────────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  Positioning Engine  │
                        │  (NLOS-aware)        │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   Final Position     │
                        │   (x, y, z)          │
                        └──────────────────────┘
```

---

## 2. Detailed LNN Architecture (Refined)

### 2.1 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INPUT PROCESSING STAGE                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
    ┌───────────▼────────────┐                 ┌───────────▼────────────┐
    │  Raw CIR Sequence      │                 │  Context Extractor     │
    │  Shape: (B, 1016, 1)   │                 │  (Domain Knowledge)    │
    │                        │                 │                        │
    │  • Amplitude samples   │                 │  Computes:             │
    │  • 15.65 ps resolution │                 │  • Rise_Time_ns        │
    │  • 15.9 ns total       │                 │  • RiseRatio           │
    └───────────┬────────────┘                 │  • E_tail              │
                │                              │  • Peak_SNR            │
                │                              │  • multipath_count     │
                │                              └───────────┬────────────┘
                │                                          │
                │                         ┌────────────────▼────────────────┐
                │                         │  Context Vector (B, 7)          │
                │                         │  [Normalized to [0,1]]          │
                │                         └────────────────┬────────────────┘
                │                                          │
┌───────────────┴──────────────────────────────────────────┴───────────────────┐
│                      LIQUID NEURAL NETWORK ENCODER                            │
│                      (Multi-Scale Temporal Processing)                        │
└───────────────────────────────────────────────────────────────────────────────┘
                │                                          │
                │                                          │
    ┌───────────▼───────────┐                             │
    │                       │              Context Modulation (Parallel)
    │   CIR Sequence        │                             │
    │   Broadcast to        │              ┌──────────────┼──────────────┐
    │   3 Parallel Paths    │              │              │              │
    │                       │              │              │              │
    └───┬───────┬───────┬───┘              │              │              │
        │       │       │                  │              │              │
  ┌─────▼────┐  │       │                  │              │              │
  │ Small-τ  │  │       │                  │              │              │
  │  Layer   │  │       │        ┌─────────▼─────────┐    │              │
  │          │  │       │        │  τ_gate_small     │    │              │
  │ Base:    │  │       │        │  = σ(W_s·ctx+b_s) │    │              │
  │ 50 ps    │◄─┼───────┼────────┤  Range: [0,1]     │    │              │
  │          │  │       │        │  ↓                 │    │              │
  │ Target:  │  │       │        │  τ_eff = 50ps ×   │    │              │
  │ Rise     │  │       │        │  (0.5+1.5*gate)   │    │              │
  │ dynamics │  │       │        └───────────────────┘    │              │
  └────┬─────┘  │       │                                 │              │
       │        │       │                                 │              │
       │   ┌────▼────┐  │                      ┌──────────▼──────────┐   │
       │   │Medium-τ │  │                      │  τ_gate_medium      │   │
       │   │ Layer   │  │                      │  = σ(W_m·ctx+b_m)   │   │
       │   │         │  │                      │  Range: [0,1]       │   │
       │   │ Base:   │◄─┼──────────────────────┤  ↓                  │   │
       │   │ 1 ns    │  │                      │  τ_eff = 1ns ×      │   │
       │   │         │  │                      │  (0.5+1.5*gate)     │   │
       │   │ Target: │  │                      └─────────────────────┘   │
       │   │ First   │  │                                                │
       │   │ bounce  │  │                                                │
       │   └────┬────┘  │                               ┌────────────────▼────┐
       │        │       │                               │  τ_gate_large       │
       │        │  ┌────▼────┐                          │  = σ(W_l·ctx+b_l)   │
       │        │  │ Large-τ │                          │  Range: [0,1]       │
       │        │  │  Layer  │                          │  ↓                  │
       │        │  │         │◄─────────────────────────┤  τ_eff = 5ns ×      │
       │        │  │ Base:   │                          │  (0.5+1.5*gate)     │
       │        │  │ 5 ns    │                          └─────────────────────┘
       │        │  │         │
       │        │  │ Target: │
       │        │  │ Multi-  │
       │        │  │ path    │
       │        │  │ tail    │
       │        │  └────┬────┘
       │        │       │
       └────────┼───────┘
                │
    ┌───────────▼──────────┐
    │                      │
    │  Extract Final       │
    │  Hidden States       │
    │  (t = 1016)          │
    │                      │
    │  h_small:  (B, 64)   │
    │  h_medium: (B, 64)   │
    │  h_large:  (B, 64)   │
    └───────────┬──────────┘
                │
    ┌───────────▼──────────┐
    │   Feature Fusion     │
    │   (Concatenation)    │
    │                      │
    │   h_fused: (B, 192)  │
    └───────────┬──────────┘
                │
┌───────────────┴───────────────┐
│      OUTPUT HEADS STAGE       │
└───────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
┌───────▼────────┐  ┌───▼──────────────┐
│ Classification │  │   Regression     │
│ Head (FCN)     │  │   Head (FCN)     │
│                │  │                  │
│ Linear(192,64) │  │ Linear(192,64)   │
│ → ReLU         │  │ → ReLU           │
│ → Dropout(0.3) │  │ → Dropout(0.3)   │
│ → Linear(64,1) │  │ → Linear(64,1)   │
│ → Sigmoid      │  │ → Linear         │
└───────┬────────┘  └───┬──────────────┘
        │               │
┌───────▼────────┐  ┌───▼──────────────┐
│  P(NLOS)       │  │  Distance        │
│  [0, 1]        │  │  Correction (m)  │
└────────────────┘  └──────────────────┘
```

### 2.2 Context-LTC Cell Internal Structure (Zoomed In)

```
┌─────────────────────────────────────────────────────────────────────────┐
│               ContextLTCCell at Timestep t                              │
│               (Executes 1,016 times per sample)                         │
└─────────────────────────────────────────────────────────────────────────┘

Inputs at time t:
  • CIR(t): Scalar amplitude                    [External, from raw data]
  • x_{t-1}: Previous hidden state (B, H)       [Internal, from t-1]
  • Context: Feature vector (B, 7)              [External, constant per sample]

┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Tau Modulation (Context-Driven)                                │
└──────────────────────────────────────────────────────────────────────────┘
    Context (B,7)
         │
         │
    ┌────▼──────────────┐
    │ Linear_tau_mod    │  W_gate: (7, 1), b_gate: (1,)
    │ W·ctx + b         │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │   Sigmoid         │  Maps to [0, 1]
    └────┬──────────────┘
         │ tau_gate ∈ [0,1]
         │
    ┌────▼──────────────┐
    │  Scale Transform  │  modulation = 0.5 + 1.5 * tau_gate
    └────┬──────────────┘
         │ modulation ∈ [0.5, 2.0]
         │
    ┌────▼──────────────┐
    │  Multiply Base τ  │  τ_current = τ_base × modulation
    └────┬──────────────┘
         │
         │ τ_current (dynamic per sample)
         │
         └──────────────────────────────────┐
                                            │
┌──────────────────────────────────────────┼────────────────────────────┐
│  STEP 2: LTC Dynamics (ODE Integration)  │                            │
└──────────────────────────────────────────┼────────────────────────────┘
                                            │
    CIR(t) ──┐                             │
             │                              │
    ┌────────▼──────────┐                  │
    │  Linear_input     │  W_in            │
    │  W_in · CIR(t)    │                  │
    └────────┬──────────┘                  │
             │                              │
             ├──────────────┐               │
             │              │               │
    x_{t-1} ─┤              │               │
             │              │               │
    ┌────────▼──────────┐  │               │
    │  Linear_recurrent │  │               │
    │  W_rec · x_{t-1}  │  │               │
    └────────┬──────────┘  │               │
             │              │               │
    ┌────────▼──────────┐  │               │
    │  Add & Tanh       │  │               │
    │  f = tanh(...)    │  │               │
    └────────┬──────────┘  │               │
             │              │               │
    ┌────────▼──────────┐  │               │
    │  Compute dx/dt    │◄─┼───────────────┘ τ_current
    │                   │  │
    │  dx/dt = (-x_{t-1} + f) / τ_current   │
    │                   │  │
    └────────┬──────────┘  │
             │              │
    ┌────────▼──────────┐  │
    │  Euler Integration│  │
    │  x_t = x_{t-1} +  │  │
    │        dt·(dx/dt) │  │
    └────────┬──────────┘  │
             │              │
             │              │
    ┌────────▼──────────┐  │
    │  New Hidden State │  │
    │  x_t  (B, H)      │  │
    └────────┬──────────┘  │
             │              │
             └──────────────┘
                   │
           Output: x_t (to next timestep or final output)
```

---

## 3. Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Raw Dataset     │  4,000 samples (2,000 LOS + 2,000 NLOS)
│  merged_cir.csv  │  1,028 columns (FP_INDEX, CIR0-CIR1015, etc.)
└────────┬─────────┘
         │
         │ Load & Validate
         │
┌────────▼─────────┐
│ Data Splitting   │  80% Train (3,200) / 20% Test (800)
│ (Stratified)     │  Preserve LOS/NLOS balance
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼────┐ ┌──▼─────┐
│ Train  │ │  Test  │
│  Set   │ │   Set  │
└───┬────┘ └────────┘
    │
    │ Feature Engineering Pipeline
    │
┌───▼──────────────────────────────────────┐
│  Preprocessing & Augmentation            │
├──────────────────────────────────────────┤
│  1. Raw CIR:                             │
│     • Normalize to [-1, 1]               │
│     • Optional: Gaussian noise (σ=0.01)  │
│                                          │
│  2. Context Features:                    │
│     • Compute from raw CIR               │
│     • MinMax scale to [0, 1]             │
│     • Store scaler for test set          │
│                                          │
│  3. Labels:                              │
│     • LOS = 0, NLOS = 1 (classification) │
│     • True distance (regression target)  │
└───┬──────────────────────────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  DataLoader (PyTorch)                    │
│  • Batch size: 32                        │
│  • Shuffle: True (train), False (test)   │
│  • Num workers: 4                        │
└───┬──────────────────────────────────────┘
    │
    │ Training Loop (100 epochs)
    │
┌───▼──────────────────────────────────────┐
│  Forward Pass                            │
│  • CIR sequence → LNN                    │
│  • Context features → τ modulation       │
│  • Dual outputs: P(NLOS), Distance       │
└───┬──────────────────────────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  Loss Computation                        │
│  • L_cls = BCELoss(pred_nlos, true_label)│
│  • L_reg = MSELoss(pred_dist, true_dist) │
│  • L_total = λ_cls·L_cls + λ_reg·L_reg   │
│    (λ_cls=1.0, λ_reg=0.1)                │
└───┬──────────────────────────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  Backward Pass & Optimization            │
│  • Optimizer: AdamW (lr=1e-3)            │
│  • Scheduler: ReduceLROnPlateau          │
│  • Gradient clipping: max_norm=1.0       │
└───┬──────────────────────────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  Validation (Every Epoch)                │
│  • Test set inference                    │
│  • Compute metrics:                      │
│    - Accuracy, Precision, Recall, F1     │
│    - Distance MAE, RMSE                  │
│  • Save best model (highest accuracy)    │
└───┬──────────────────────────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  Early Stopping Check                    │
│  • Patience: 15 epochs                   │
│  • Monitor: Validation accuracy          │
│  • Stop if no improvement                │
└───┬──────────────────────────────────────┘
    │
┌───▼──────────────────────────────────────┐
│  Final Model                             │
│  • best_model.pth                        │
│  • Checkpoint: epoch, weights, optimizer │
└──────────────────────────────────────────┘
```

---

## 4. Comparison with Baseline Architectures

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ARCHITECTURE COMPARISON DIAGRAM                            │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────┐
│  A. Logistic Regression│  (Baseline)
├────────────────────────┤
│  Input: 6 Features     │
│  • FP_INDEX_scaled     │  ┌──────────────┐
│  • Max_Index           │  │ StandardScaler│
│  • ROI_energy          │─→│ + LogReg      │─→ LOS/NLOS
│  • fp_peak_amp         │  └──────────────┘
│  • first_bounce_delay  │  86.8% Accuracy
│  • multipath_count     │  (Hand-crafted features only)
└────────────────────────┘

┌────────────────────────┐
│  B. Standard LSTM      │  (Temporal Baseline)
├────────────────────────┤
│  Input: CIR (1016,1)   │
│                        │  ┌──────────────┐
│  CIR0, CIR1, ...,      │  │ LSTM(64)     │
│  CIR1015               │─→│ LSTM(64)     │─→ LOS/NLOS
│                        │  │ FC(1)+Sigmoid│
│  (Sequential)          │  └──────────────┘
└────────────────────────┘  Fixed τ (single timescale)

┌────────────────────────┐
│  C. 1D CNN             │  (Spatial Baseline)
├────────────────────────┤
│  Input: CIR (1016,1)   │
│                        │  ┌──────────────┐
│  [Raw waveform]        │  │ Conv1D(32,k=5)│
│                        │─→│ MaxPool(2)    │─→ LOS/NLOS
│                        │  │ Conv1D(64,k=3)│
│  (Spatial patterns)    │  │ GlobalAvgPool │
└────────────────────────┘  └──────────────┘

┌────────────────────────┐
│  D. Multi-Scale LNN    │  ⭐ (Your Approach)
├────────────────────────┤
│  Input 1: CIR (1016,1) │
│  Input 2: Context (7)  │  ┌──────────────────────┐
│                        │  │ Small-τ LNN (50ps)   │
│  • Raw temporal data   │  │ Medium-τ LNN (1ns)   │─→ LOS/NLOS
│  • Domain knowledge    │─→│ Large-τ LNN (5ns)    │   + Distance
│  • Adaptive τ          │  │ τ ← Context features │
│                        │  └──────────────────────┘
│  (Multi-scale + adaptive) Expected: 90-95%
└────────────────────────┘

Key Advantages of Multi-Scale LNN:
✅ Adaptive temporal integration (context-guided τ)
✅ Multi-scale processing (captures all phenomena)
✅ Domain knowledge injection (physics-informed)
✅ Dual-task learning (classification + regression)
```

---

## 5. Physical Interpretation Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│          PHYSICAL SIGNAL → NEURAL NETWORK MAPPING                       │
└─────────────────────────────────────────────────────────────────────────┘

LOS Signal Characteristics:
┌────────────────────────────────────────┐
│    Amplitude                           │
│      ▲                                 │
│      │    ⬆ Sharp Peak                │
│      │   /│\                           │
│   8k │  / │ \                          │
│      │ /  │  \___                      │
│   4k │/   │      \___                  │
│      │    │          \____             │
│    0 └────┼────────────────▶ Time     │
│       t_start  t_peak                  │
│       (FP)     (Max)                   │
│                                        │
│  Rise_Time: 2.65 indices (42 ps) ⬅ Fast│
│  RiseRatio: 0.245           ⬅ Sharp    │
│  E_tail: 0.659              ⬅ Low      │
└────────────────────────────────────────┘
            ↓
      ┌──────────┐
      │ Context  │ → [Fast rise, Low tail]
      │ Features │
      └─────┬────┘
            │
            ├─→ τ_small = 50ps × 0.5 = 25ps ✓ Fast integration
            ├─→ τ_medium = 1ns × 0.7 = 0.7ns
            └─→ τ_large = 5ns × 0.8 = 4ns
            
Result: Network quickly captures sharp peak, minimal tail processing


NLOS Signal Characteristics:
┌────────────────────────────────────────┐
│    Amplitude                           │
│      ▲                                 │
│      │   Gradual Rise ⬇               │
│      │  _--│--__                       │
│   8k │_/   │    \__                    │
│      │     │       \_                  │
│   4k │     │         \___  Multipath ⬇│
│      │     │            \~~\~~\~~\_    │
│    0 └─────┼────────────────────────▶ Time
│       t_start  t_peak                  │
│       (FP)     (Max)                   │
│                                        │
│  Rise_Time: 1.57 indices (25 ps) ⬅ Slow│
│  RiseRatio: 0.327            ⬅ Gradual │
│  E_tail: 0.810               ⬅ High    │
└────────────────────────────────────────┘
            ↓
      ┌──────────┐
      │ Context  │ → [Slow rise, High tail]
      │ Features │
      └─────┬────┘
            │
            ├─→ τ_small = 50ps × 1.5 = 75ps ✓ Slower to capture gradual rise
            ├─→ τ_medium = 1ns × 1.8 = 1.8ns ✓ Integrate reflections
            └─→ τ_large = 5ns × 2.0 = 10ns ✓ Capture full tail
            
Result: Network slowly integrates dispersed energy, captures multipath
```

---

## 6. Recommended Diagram Tools

For your capstone presentation, use these tools:

### For System Architecture:
- **Draw.io** (diagrams.net) - Free, web-based, export to SVG/PNG
- **Lucidchart** - Professional, templates available
- **Microsoft Visio** - If available through university

### For Neural Network Diagrams:
- **NN-SVG** (alexlenail.me/NN-SVG) - Quick neural network visualizations
- **PlotNeuralNet** (GitHub) - LaTeX-based, publication-quality
- **Netron** - Visualize trained models

### For Equations & Math:
- **LaTeX with TikZ** - Publication quality
- **Microsoft PowerPoint** - SmartArt + Equation Editor
- **Keynote** (Mac) - Clean, modern diagrams

---

## 7. Presentation Recommendations

### Slide 1: System Overview
- Use diagram from Section 1 (high-level system)
- Emphasize: Data → LNN → Positioning

### Slide 2: LNN Architecture
- Use simplified version of Section 2.1
- Focus on two streams (CIR + Context)
- Show three tau layers

### Slide 3: Context-Driven Adaptation
- Use Section 2.2 (ContextLTCCell)
- Animate: Context → τ_gate → Dynamic τ

### Slide 4: Physical Interpretation
- Use Section 5 (LOS vs NLOS signals)
- Side-by-side comparison
- Show how context features guide τ

### Slide 5: Comparison
- Use Section 4 (Baseline comparison)
- Highlight advantages of your approach

---

## 8. Color Scheme Recommendations

For professional appearance:

**LOS Signal:** 
- Primary: `#2E7D32` (Green) - Direct, clear
- Accent: `#66BB6A` (Light green)

**NLOS Signal:**
- Primary: `#C62828` (Red) - Obstructed, complex  
- Accent: `#EF5350` (Light red)

**Neural Network Components:**
- Input layers: `#1976D2` (Blue)
- LNN layers: `#F57C00` (Orange)
- Context features: `#7B1FA2` (Purple)
- Output: `#388E3C` (Dark green)

**Background:**
- Main: `#FFFFFF` (White)
- Boxes: `#F5F5F5` (Light gray)
- Borders: `#424242` (Dark gray)

---

**Next:** See `02_Implementation_Details.md` for code implementation!
