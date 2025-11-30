# Experimental Design
## Rigorous Evaluation Protocol for Capstone

**Purpose:** Comprehensive experimental framework to validate Multi-Scale LNN superiority

---

## 1. Research Questions & Hypotheses

### 1.1 Primary Research Question

**RQ1:** Does context-guided multi-scale liquid neural network (LNN) outperform existing methods for UWB LOS/NLOS classification?

**Hypothesis H1:** Multi-Scale LNN achieves ≥92% accuracy, significantly better than:
- Logistic Regression baseline (86.8%)
- Best deep learning baseline (LSTM/Transformer ~92%)

### 1.2 Secondary Research Questions

**RQ2:** What is the contribution of context-guided tau modulation?

**Hypothesis H2:** Context features improve accuracy by ≥2% compared to fixed-tau LNN.

**RQ3:** Is multi-scale processing necessary?

**Hypothesis H3:** Three-tau architecture outperforms single-tau by ≥1.5%.

**RQ4:** Do learned tau values correlate with physical signal characteristics?

**Hypothesis H4:** LOS signals → smaller tau, NLOS signals → larger tau (validated through visualization).

---

## 2. Experimental Setup

### 2.1 Dataset Configuration

```
Total Samples: 4,000
├── LOS: 2,000 (50%)
│   ├── 2m living room: 1,000
│   └── 4.3m corner: 1,000
└── NLOS: 2,000 (50%)
    ├── 1.56m open door: 1,000
    └── 4.4m closed door: 1,000

Split Strategy:
├── Train: 3,200 (80%) - Stratified by label
├── Test: 800 (20%) - Stratified by label
└── Random Seed: 42 (reproducibility)
```

### 2.2 Cross-Validation Strategy

**Primary Evaluation: Holdout Test Set**
- Train on 3,200 samples
- Test on 800 samples (never seen during training)
- Report final metrics on this set

**Robustness Check: 5-Fold Cross-Validation**
- Stratified K-Fold (k=5)
- Report mean ± std across folds
- Ensure model generalizes beyond single split

**Generalization Test: Leave-One-Scenario-Out (LOSO)**
```
Fold 1: Train on [LOS 2m, LOS 4.3m, NLOS 1.56m], Test on [NLOS 4.4m]
Fold 2: Train on [LOS 2m, LOS 4.3m, NLOS 4.4m], Test on [NLOS 1.56m]
Fold 3: Train on [LOS 2m, NLOS 1.56m, NLOS 4.4m], Test on [LOS 4.3m]
Fold 4: Train on [LOS 4.3m, NLOS 1.56m, NLOS 4.4m], Test on [LOS 2m]
```
- Tests if model generalizes to unseen scenarios
- Critical for real-world deployment validation

---

## 3. Experimental Pipeline

### 3.1 Experiment Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                   EXPERIMENTAL PIPELINE                        │
└────────────────────────────────────────────────────────────────┘

STAGE 1: Data Preprocessing
├── Load merged_cir.csv
├── Extract context features (7 features)
├── Normalize CIR to [-1, 1]
├── Scale context to [0, 1]
└── Split: 80/20 train/test

STAGE 2: Baseline Experiments
├── Experiment 2.1: Logistic Regression (Already done ✅)
├── Experiment 2.2: Random Forest
├── Experiment 2.3: XGBoost
├── Experiment 2.4: 1D CNN
├── Experiment 2.5: LSTM
├── Experiment 2.6: Bi-LSTM
└── Experiment 2.7: Transformer

STAGE 3: Proposed Model Experiments
├── Experiment 3.1: Single-Tau LNN (ablation)
├── Experiment 3.2: Two-Tau LNN (ablation)
├── Experiment 3.3: Multi-Scale LNN (no context)
└── Experiment 3.4: Multi-Scale LNN (full) 🎯

STAGE 4: Ablation Studies
├── Ablation 4.1: Remove context modulation
├── Ablation 4.2: Remove multi-scale (single tau)
├── Ablation 4.3: Random context features
└── Ablation 4.4: No context normalization

STAGE 5: Sensitivity Analysis
├── Sensitivity 5.1: Vary tau_small [25ps, 50ps, 100ps]
├── Sensitivity 5.2: Vary tau_medium [0.5ns, 1ns, 2ns]
├── Sensitivity 5.3: Vary tau_large [2.5ns, 5ns, 10ns]
└── Sensitivity 5.4: Vary hidden_size [32, 64, 128]

STAGE 6: Statistical Validation
├── McNemar's test (paired predictions)
├── Cross-validation (5-fold stratified)
├── LOSO validation (scenario generalization)
└── Bootstrap confidence intervals (1000 iterations)

STAGE 7: Analysis & Visualization
├── ROC curves comparison
├── Confusion matrices
├── Feature importance (for baselines)
├── Tau modulation analysis (for LNN)
├── Error case analysis
└── Computational efficiency comparison
```

---

## 4. Detailed Experiment Specifications

### 4.1 Baseline Experiments

#### Experiment 2.1: Logistic Regression ✅ (Done)

```yaml
Experiment ID: EXP-2.1
Model: Logistic Regression
Features: [FP_INDEX_scaled, Max_Index, roi_energy, fp_peak_amp, 
           first_bounce_delay_ns, multipath_count]
Preprocessing: StandardScaler
Hyperparameters:
  - solver: lbfgs
  - max_iter: 1000
  - random_state: 42
Expected Result: 86.8% (already achieved)
Purpose: Establish baseline performance
```

#### Experiment 2.4: 1D CNN

```yaml
Experiment ID: EXP-2.4
Model: 1D Convolutional Neural Network
Input: Raw CIR (1016, 1)
Architecture:
  - Conv1D(32, k=5) → BatchNorm → ReLU → MaxPool(2)
  - Conv1D(64, k=3) → BatchNorm → ReLU → MaxPool(2)
  - Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)
  - AdaptiveAvgPool → FC(64) → Dropout(0.3) → FC(1) → Sigmoid
Hyperparameters:
  - Batch size: 32
  - Learning rate: 1e-3
  - Optimizer: AdamW
  - Epochs: 100
  - Early stopping: patience=15
Expected Result: 88-92%
Purpose: Test spatial pattern learning
```

#### Experiment 2.5: LSTM

```yaml
Experiment ID: EXP-2.5
Model: Long Short-Term Memory
Input: Raw CIR (1016, 1)
Architecture:
  - LSTM(64, num_layers=2, dropout=0.3)
  - FC(64) → ReLU → Dropout(0.3) → FC(1) → Sigmoid
Hyperparameters:
  - Batch size: 32
  - Learning rate: 1e-3
  - Optimizer: AdamW
  - Epochs: 100
  - Gradient clipping: max_norm=1.0
Expected Result: 89-93%
Purpose: Test fixed temporal integration
```

### 4.2 Proposed Model Experiments

#### Experiment 3.4: Multi-Scale LNN (Full Model) 🎯

```yaml
Experiment ID: EXP-3.4
Model: Context-Guided Multi-Scale Liquid Neural Network
Inputs:
  - Raw CIR: (B, 1016, 1)
  - Context Features: (B, 7)
Architecture:
  Small-Tau Layer:
    - tau_base: 50e-12 s (50 ps)
    - hidden_size: 64
    - tau_range: [0.5, 2.0]
  Medium-Tau Layer:
    - tau_base: 1e-9 s (1 ns)
    - hidden_size: 64
    - tau_range: [0.5, 2.0]
  Large-Tau Layer:
    - tau_base: 5e-9 s (5 ns)
    - hidden_size: 64
    - tau_range: [0.5, 2.0]
  Fusion & Output:
    - Concat(3×64) → FC(64) → ReLU → Dropout(0.3) → FC(1) → Sigmoid
Hyperparameters:
  - Batch size: 32
  - Learning rate: 1e-3
  - Optimizer: AdamW(weight_decay=1e-4)
  - Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
  - Epochs: 100
  - Early stopping: patience=15
  - Gradient clipping: max_norm=1.0
  - Loss weights: λ_cls=1.0, λ_reg=0.1
Expected Result: 92-95%
Purpose: Main proposed method
```

### 4.3 Ablation Experiments

#### Ablation 4.1: No Context Modulation

```yaml
Experiment ID: ABL-4.1
Variant: Fixed tau (no context features)
Modification: Remove tau_gate, use fixed tau values
Expected Delta: -2.5% accuracy
Purpose: Validate contribution of context-guided modulation
```

#### Ablation 4.2: Single-Tau Only

```yaml
Experiment ID: ABL-4.2
Variant: Single LTC layer (tau=1ns, hidden=192)
Modification: Remove multi-scale architecture
Expected Delta: -2.0% accuracy
Purpose: Validate benefit of multi-scale processing
```

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

**Classification Metrics:**
```python
metrics = {
    'Accuracy': (TP + TN) / (TP + TN + FP + FN),
    'Precision': TP / (TP + FP),  # Positive Predictive Value
    'Recall': TP / (TP + FN),     # Sensitivity, True Positive Rate
    'Specificity': TN / (TN + FP), # True Negative Rate
    'F1-Score': 2 * (Precision * Recall) / (Precision + Recall),
    'AUC-ROC': area_under_roc_curve
}
```

**Distance Regression Metrics (Secondary Task):**
```python
regression_metrics = {
    'MAE': mean_absolute_error(y_true_dist, y_pred_dist),
    'RMSE': sqrt(mean_squared_error(y_true_dist, y_pred_dist)),
    'R²': r2_score(y_true_dist, y_pred_dist)
}
```

### 5.2 Computational Metrics

```python
efficiency_metrics = {
    'Parameters': count_parameters(model),
    'Training Time': time_to_train_100_epochs,
    'Inference Time': average_time_per_sample,
    'GPU Memory': peak_gpu_memory_usage,
    'FLOPs': floating_point_operations
}
```

### 5.3 Interpretability Metrics

**For LNN Models:**
```python
interpretability = {
    'Tau Correlation': correlation(tau_values, [LOS/NLOS]),
    'Context Importance': gradient_based_feature_importance,
    'Tau Consistency': std(tau_values_per_class)
}
```

---

## 6. Statistical Validation

### 6.1 Significance Testing

**McNemar's Test (Paired Predictions):**
```python
"""
Test if Multi-Scale LNN significantly outperforms baseline
H0: Both models have same error rate
H1: Multi-Scale LNN has lower error rate
"""
from scipy.stats import mcnemar

# Contingency table
table = [[n00, n01],  # n00: both wrong, n01: baseline correct, LNN wrong
         [n10, n11]]  # n10: baseline wrong, LNN correct, n11: both correct

result = mcnemar(table, exact=True)
if result.pvalue < 0.05:
    print("✅ Multi-Scale LNN significantly better!")
```

**Bootstrap Confidence Intervals:**
```python
"""
95% confidence intervals for accuracy
"""
def bootstrap_ci(y_true, y_pred, n_iterations=1000, alpha=0.05):
    np.random.seed(42)
    accuracies = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        acc = accuracy_score(y_true_boot, y_pred_boot)
        accuracies.append(acc)
    
    lower = np.percentile(accuracies, alpha/2 * 100)
    upper = np.percentile(accuracies, (1 - alpha/2) * 100)
    
    return (lower, upper)

# Example: Multi-Scale LNN accuracy = 93.5% [92.8%, 94.2%] (95% CI)
```

### 6.2 Cross-Validation Protocol

```python
"""
5-Fold Stratified Cross-Validation
"""
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/5")
    
    # Train model
    model.fit(X_train[train_idx], y_train[train_idx])
    
    # Validate
    acc = model.score(X_train[val_idx], y_train[val_idx])
    fold_results.append(acc)

mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)

print(f"CV Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
```

---

## 7. Result Reporting Template

### 7.1 Main Results Table

```markdown
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Training Time | Inference (ms) |
|-------|----------|-----------|--------|-------|---------|---------------|----------------|
| Logistic Regression | 86.8% ± 0.5% | 85.7% | 88.2% | 86.9% | 0.920 | < 1s | 0.1 |
| Random Forest | 88.5% ± 0.8% | 87.2% | 89.5% | 88.3% | 0.940 | 10s | 2.0 |
| XGBoost | 90.2% ± 0.6% | 89.1% | 91.0% | 90.0% | 0.960 | 30s | 3.0 |
| 1D CNN | 90.5% ± 1.2% | 89.5% | 91.2% | 90.3% | 0.950 | 5min | 5.0 |
| LSTM | 91.8% ± 0.9% | 90.8% | 92.5% | 91.6% | 0.970 | 15min | 8.0 |
| Transformer | 92.8% ± 1.0% | 92.0% | 93.2% | 92.6% | 0.975 | 25min | 15.0 |
| **Multi-Scale LNN** | **93.5% ± 0.7%** | **92.8%** | **94.0%** | **93.4%** | **0.980** | 20min | 18.0 |

*Note: Accuracy reported as Mean ± Std from 5-fold CV*
```

### 7.2 Ablation Study Results

```markdown
| Variant | Accuracy | Δ from Full | Statistical Sig. |
|---------|----------|-------------|------------------|
| Full Multi-Scale LNN | 93.5% | Baseline | - |
| No Context Modulation | 91.0% | **-2.5%** | p < 0.01 ✅ |
| Single-Tau (1ns) | 91.5% | **-2.0%** | p < 0.05 ✅ |
| Two-Tau (50ps, 5ns) | 92.8% | **-0.7%** | p = 0.08 |
| Random Context | 88.0% | **-5.5%** | p < 0.001 ✅ |
```

### 7.3 LOSO Validation Results

```markdown
| Test Scenario | Train Scenarios | Accuracy | Comment |
|---------------|----------------|----------|---------|
| NLOS 4.4m closed | LOS 2m, LOS 4.3m, NLOS 1.56m | 91.2% | Hardest scenario |
| NLOS 1.56m open | LOS 2m, LOS 4.3m, NLOS 4.4m | 93.8% | Good generalization |
| LOS 4.3m corner | LOS 2m, NLOS 1.56m, NLOS 4.4m | 94.5% | LOS easier |
| LOS 2m living | LOS 4.3m, NLOS 1.56m, NLOS 4.4m | 94.2% | LOS easier |
| **Average** | - | **93.4%** | ✅ Generalizes well |
```

---

## 8. Visualization Requirements

### 8.1 Essential Figures

**Figure 1: ROC Curve Comparison**
- All models on same plot
- AUC values in legend
- 45° diagonal reference line

**Figure 2: Confusion Matrices (Side-by-side)**
- 2×3 grid showing top 6 models
- Heatmaps with annotations
- Normalize by true class

**Figure 3: Training Curves**
- Loss curves (train & validation)
- Accuracy curves (train & validation)
- Show early stopping point

**Figure 4: Ablation Study Bar Chart**
- Accuracy for each variant
- Error bars (±1 std)
- Highlight significant drops

**Figure 5: Tau Modulation Analysis**
- Boxplot: Tau values for LOS vs NLOS
- Separate for small/medium/large tau
- Show context feature correlation

**Figure 6: Error Case Analysis**
- Sample CIR waveforms of:
  - Correct LOS predictions
  - Correct NLOS predictions
  - False Positives (LOS → NLOS)
  - False Negatives (NLOS → LOS)

### 8.2 Supplementary Visualizations

**Figure S1: Context Feature Distributions**
- Violin plots for 7 features
- LOS vs NLOS comparison
- Show overlap regions

**Figure S2: Tau Sensitivity Analysis**
- Heatmap: Accuracy vs (tau_small, tau_large)
- Identify optimal region

**Figure S3: Feature Importance (Baselines)**
- Bar chart for Random Forest/XGBoost
- Compare with logistic regression coefficients

---

## 9. Reproducibility Checklist

### 9.1 Code & Environment

```yaml
Environment:
  - Python: 3.8+
  - PyTorch: 2.0+
  - CUDA: 11.8 (if GPU available)
  - Dependencies: requirements.txt

Random Seeds:
  - Data split: 42
  - Model initialization: 42
  - Cross-validation: 42
  - NumPy: np.random.seed(42)
  - PyTorch: torch.manual_seed(42)

Code Structure:
  - All code in Git repository
  - Each experiment has separate config file
  - Results logged to MLflow/Weights&Biases
  - Model checkpoints saved with metadata
```

### 9.2 Documentation

```markdown
Required Documentation:
1. README.md with setup instructions
2. EXPERIMENTS.md listing all experiments
3. RESULTS.md with tables and figures
4. requirements.txt with exact versions
5. Dockerfile for containerized reproduction
6. Jupyter notebooks for analysis
7. Thesis chapter with detailed methodology
```

---

## 10. Timeline & Milestones

### 10.1 Execution Schedule

```
Week 1-2: Baseline Experiments
├── ✅ Logistic Regression (Already done)
├── Day 1-2: Random Forest & XGBoost
├── Day 3-5: 1D CNN
├── Day 6-8: LSTM & Bi-LSTM
└── Day 9-10: Transformer

Week 3: Proposed Model
├── Day 1-3: Implement Multi-Scale LNN
├── Day 4-5: Hyperparameter tuning
└── Day 6-7: Training & evaluation

Week 4: Ablation & Sensitivity
├── Day 1-3: Ablation experiments
├── Day 4-5: Sensitivity analysis
└── Day 6-7: Statistical validation

Week 5: Analysis & Visualization
├── Day 1-2: Generate all figures
├── Day 3-4: Error case analysis
├── Day 5-6: Write results section
└── Day 7: Final review & polish

Week 6: Thesis Writing
├── Day 1-3: Methodology chapter
├── Day 4-5: Results chapter
└── Day 6-7: Discussion & conclusion
```

### 10.2 Deliverables

```
Deliverable 1 (Week 2): Baseline Results Report
├── Performance table for all baselines
├── ROC curves
└── Initial comparison

Deliverable 2 (Week 3): Main Results
├── Multi-Scale LNN performance
├── Comparison with baselines
└── Statistical significance tests

Deliverable 3 (Week 4): Ablation & Sensitivity
├── Ablation study results
├── Sensitivity analysis
└── Optimal hyperparameters

Deliverable 4 (Week 5): Final Figures & Analysis
├── All publication-ready figures
├── Error case analysis
└── Interpretation of results

Deliverable 5 (Week 6): Thesis Chapter
├── Complete experimental section
├── Results & discussion
└── Ready for supervisor review
```

---

## 11. Risk Mitigation

### 11.1 Potential Issues & Solutions

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Multi-Scale LNN doesn't outperform baselines** | Low | High | • Tune hyperparameters extensively<br>• Try different tau values<br>• Increase model capacity |
| **Training instability (exploding gradients)** | Medium | Medium | • Gradient clipping (max_norm=1.0)<br>• Smaller learning rate<br>• Better initialization |
| **Overfitting on small dataset** | Medium | Medium | • Strong regularization (dropout=0.3)<br>• Data augmentation (noise)<br>• Early stopping |
| **Long training time** | High | Low | • Use GPU acceleration<br>• Reduce sequence length if needed<br>• Start early! |
| **Context features not discriminative** | Low | High | • Already validated in EDA (✅)<br>• Try additional features if needed |

### 11.2 Backup Plans

**If Multi-Scale LNN underperforms:**
1. Fallback to Two-Tau architecture (simpler, still novel)
2. Focus thesis on analysis of why multi-scale helps (ablation study)
3. Emphasize interpretability benefits (tau modulation visualization)

**If training takes too long:**
1. Use smaller hidden size (32 instead of 64)
2. Train on subset of data first (proof of concept)
3. Parallelize experiments across multiple GPUs

---

## 12. Success Criteria

### 12.1 Minimum Viable Results

✅ **Must Achieve:**
- Multi-Scale LNN accuracy ≥ 90% (above XGBoost baseline)
- Statistically significant improvement over LSTM (p < 0.05)
- Context modulation shows clear LOS/NLOS tau difference

✅ **Should Achieve:**
- Multi-Scale LNN accuracy ≥ 92% (above Transformer)
- Ablation studies show each component contributes ≥1%
- LOSO validation accuracy ≥ 90%

🎯 **Stretch Goals:**
- Multi-Scale LNN accuracy ≥ 95%
- Outperform Transformer by ≥1.5%
- Publish paper at conference

### 12.2 Thesis Evaluation Rubric

**Experimental Rigor (40%):**
- Comprehensive baseline comparison ✅
- Proper train/test split & cross-validation ✅
- Statistical significance testing ✅
- Ablation studies ✅

**Technical Innovation (30%):**
- Novel architecture (multi-scale LNN) ✅
- Domain knowledge integration (context features) ✅
- Theoretical foundation (adaptive tau) ✅

**Results & Analysis (20%):**
- Clear improvement over baselines ⏳ (To demonstrate)
- Insightful error analysis ⏳
- Interpretability analysis ⏳

**Documentation (10%):**
- Clear methodology ✅
- Reproducible experiments ⏳
- Professional figures ⏳

---

## Summary

This experimental design provides a **comprehensive, rigorous framework** to validate your Multi-Scale LNN approach. Key strengths:

1. ✅ **Comprehensive Baselines** - 7 different models for comparison
2. ✅ **Statistical Rigor** - McNemar's test, cross-validation, bootstrap CI
3. ✅ **Thorough Ablation** - Test each component's contribution
4. ✅ **Reproducibility** - Fixed seeds, documented hyperparameters
5. ✅ **Clear Timeline** - 6-week execution plan

**Your next steps:**
1. Start with baseline experiments (Week 1-2)
2. Implement Multi-Scale LNN (Week 3)
3. Run ablation studies (Week 4)
4. Generate visualizations (Week 5)
5. Write thesis chapter (Week 6)

**Good luck with your capstone! You have a strong foundation.** 🚀
