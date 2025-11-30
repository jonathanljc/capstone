# Exploratory Data Analysis Report
## UWB Localization: LOS/NLOS Classification

**Author:** [Lim Jing Chuan Jonathan 2300923]  
**Date:** November 30, 2025  
**Project:** UWB Indoor Localization System

---

## Executive Summary

This report presents a comprehensive exploratory data analysis (EDA) of Ultra-Wideband (UWB) Channel Impulse Response (CIR) signals collected in indoor environments. The primary objective is to characterize the differences between Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) conditions for developing accurate indoor positioning systems.

**Key Findings:**
- Dataset comprises 4,000 measurements (2,000 LOS, 2,000 NLOS) across 4 scenarios
- NLOS signals exhibit **27.8% more multipath components** compared to LOS (17.4 vs 13.6 peaks)
- First bounce delay is **16.8% longer in NLOS** conditions (0.132 ns vs 0.113 ns)
- Baseline logistic regression classifier achieves **86.8% accuracy** in LOS/NLOS classification
- Hardware first-path index (FP_INDEX) and peak index are the most discriminative features

---

## 1. Introduction

### 1.1 Background

Ultra-Wideband (UWB) technology has emerged as a promising solution for precise indoor localization. However, NLOS conditions caused by obstructions (walls, doors, furniture) introduce significant positioning errors. Accurate LOS/NLOS identification is critical for mitigating these errors.

### 1.2 Dataset Description

**Data Collection Setup:**
- **Hardware:** DW1000 UWB transceiver
- **Environment:** Indoor residential setting (living room)
- **Measurement Type:** Channel Impulse Response (CIR) waveforms
- **Sampling:** 1,016 samples per CIR at 15.65 ps resolution

**Scenarios:**
1. **LOS 2m living room** (1,000 samples) - Clear line-of-sight, 2.0m separation
2. **LOS 4.3m corner** (1,000 samples) - Clear line-of-sight, 4.3m separation  
3. **NLOS 1.56m open door** (1,000 samples) - Obstructed path through open door, 1.56m
4. **NLOS 4.4m closed door** (1,000 samples) - Obstructed path through closed door, 4.4m

**Total Measurements:** 4,000 (balanced: 50% LOS, 50% NLOS)

---

## 2. Data Quality Assessment

### 2.1 Completeness Check

✓ **No missing values detected** across all 4,000 measurements and 1,016 CIR columns.

### 2.2 Label Distribution

The dataset is **perfectly balanced**:
- **LOS:** 2,000 samples (50%)
- **NLOS:** 2,000 samples (50%)

This balanced distribution eliminates class imbalance concerns for machine learning models.

---

## 3. Feature Engineering

### 3.1 Derived Features

To extract meaningful information from raw CIR waveforms, the following features were engineered:

| Feature | Description | Calculation Method |
|---------|-------------|-------------------|
| **ROI Energy** | Total signal energy in Region of Interest (indices 700-820) | Sum of squared amplitudes |
| **FP_INDEX_scaled** | Hardware-detected first path index | FP_INDEX ÷ 64 (register to CIR index conversion) |
| **True_Index** | Expected CIR index based on true distance | ToF = d_true / c_air; Index = ToF / TS_DW1000 |
| **Max_Index** | CIR index with maximum amplitude | argmax(CIR) |
| **Index Errors** | Deviations from true index | FP_INDEX_scaled - True_Index |
| **fp_peak_idx** | Detected first path peak location | Peak detection near hardware FP_INDEX |
| **fp_peak_amp** | First path peak amplitude | Amplitude at fp_peak_idx |
| **first_bounce_idx** | First multipath bounce location | Next significant peak after first path |
| **first_bounce_delay_ns** | Time delay to first bounce | (first_bounce_idx - fp_peak_idx) × TS_DW1000 |
| **multipath_count** | Number of detected multipath components | Peak count above 5× noise floor |

### 3.2 Constants & Parameters

- **TS_DW1000:** 15.65 ps/sample (DW1000 time resolution)
- **C_AIR:** 299,792,458 m/s (speed of light)
- **ROI_START, ROI_END:** 700-820 (empirically determined signal region)
- **Noise Threshold:** 5× median noise floor

---

## 4. Signal Characteristics Analysis

### 4.1 ROI Energy Distribution

**Observation:** LOS and NLOS signals show **overlapping but distinguishable** energy distributions in the Region of Interest (700-820 index range).

**Key Findings:**
- Both LOS and NLOS exhibit wide energy ranges due to distance variation (1.56m - 4.4m)
- Energy alone is insufficient for reliable LOS/NLOS classification
- Energy patterns suggest distance-dependent normalization may improve classification

### 4.2 Distance Error Analysis

**Observation:** NLOS conditions introduce **systematic positive bias** in distance estimation.

**Impact:**
- LOS measurements show **tighter clustering** around true distance
- NLOS measurements exhibit **larger variance and positive offset**
- Distance error magnitude correlates with obstruction severity (closed door > open door)

### 4.3 CIR Waveform Characteristics

**Mean CIR Shape Analysis (ROI: 700-820):**

| Characteristic | LOS | NLOS |
|---------------|-----|------|
| **Peak Sharpness** | Sharp, narrow peak | Broader, more dispersed |
| **Signal Decay** | Rapid fall-off after peak | Gradual decay with oscillations |
| **Background Noise** | Lower relative to peak | Higher relative to peak |
| **Waveform Stability** | Low variance (mean ± std tight) | Higher variance (mean ± std wider) |

**Interpretation:**
- LOS signals have **cleaner, more direct propagation paths**
- NLOS signals exhibit **multipath interference** causing signal spreading and prolonged tails

---

## 5. Multipath Analysis

### 5.1 Peak Detection Methodology

**Algorithm:** Simple local maxima detection with adaptive thresholding
- **Threshold:** 5× median noise floor (calculated from indices 0-600)
- **Minimum Peak Separation:** 2 indices (~31.3 ps)
- **Search Region:** 650-900 indices

### 5.2 Multipath Statistics

| Metric | LOS | NLOS | Difference |
|--------|-----|------|------------|
| **Mean Multipath Count** | 13.60 | 17.38 | +27.8% |
| **Mean First Bounce Delay (ns)** | 0.113 | 0.132 | +16.8% |
| **First Bounce Delay Distribution** | Concentrated at ~0.10 ns | Spread between 0.18-0.22 ns | More dispersed |

### 5.3 Key Observations

1. **Multipath Component Count:**
   - NLOS signals have **significantly more detected peaks** (27.8% increase)
   - Indicates more complex propagation environment with multiple reflection paths

2. **First Bounce Timing:**
   - NLOS first bounces arrive **later** relative to first path
   - Histogram shows **bimodal distribution** for NLOS vs. unimodal for LOS
   - Suggests different reflection mechanisms (wall penetration vs. simple reflection)

3. **Spatial Consistency:**
   - Multipath patterns are **scenario-dependent**
   - Closed door scenario shows highest multipath count
   - Open door scenario shows intermediate behavior

---

## 6. Detailed Waveform Visualization

### 6.1 Sample Overlays with Peak Detection

**Visualization Window:** Indices 740-790 (zoomed ROI for clarity)

**Reference Lines:**
- **Red dotted:** FP hardware index (DW1000 chip detection)
- **Green dashed:** First path peak (algorithm detection)
- **Purple dashed:** First bounce peak (first multipath reflection)

### 6.2 Scenario-Specific Patterns

#### LOS 2m Living Room
- **Delta FP→FB:** ~6.0 indices (~0.09 ns)
- **Median Peak Count:** 9 peaks
- **Characteristics:** Tight waveform clustering, consistent peak alignment

#### LOS 4.3m Corner
- **Delta FP→FB:** ~6.0 indices (~0.09 ns)
- **Median Peak Count:** 9 peaks
- **Characteristics:** Similar to 2m case with slightly lower SNR

#### NLOS 1.56m Open Door
- **Delta FP→FB:** ~8.0 indices (~0.13 ns)
- **Median Peak Count:** 9 peaks
- **Characteristics:** Wider waveform spread, visible secondary peak cluster

#### NLOS 4.4m Closed Door
- **Delta FP→FB:** ~8.0 indices (~0.13 ns)
- **Median Peak Count:** 12 peaks
- **Characteristics:** Most dispersed waveforms, strongest multipath contamination

### 6.3 Waveform Stability Analysis

**Signal Stability (Mean ± Std Dev):**

| Scenario | Peak Index | Peak Amplitude | Std Dev Observation |
|----------|-----------|----------------|---------------------|
| **LOS 2m** | 753 | ~7,500 | Narrow confidence band |
| **LOS 4.3m** | 761 | ~7,800 | Narrow confidence band |
| **NLOS 1.56m** | 755 | ~7,600 | Wider confidence band |
| **NLOS 4.4m** | 763 | ~7,000 | Widest confidence band |

**Key Finding:** NLOS signals exhibit **higher temporal variance** (wider ±1σ bands), indicating less stable channel conditions.

---

## 7. Baseline Classification Model

### 7.1 Model Architecture

**Algorithm:** Logistic Regression with Standard Scaling  
**Feature Set:** 6 engineered features
- `roi_energy`: Signal energy in ROI
- `fp_peak_amp`: First path peak amplitude  
- `first_bounce_delay_ns`: Time to first bounce
- `multipath_count`: Number of detected peaks
- `FP_INDEX_scaled`: Hardware first path index
- `Max_Index`: Index of maximum amplitude

**Training Setup:**
- **Train/Test Split:** 80/20 (stratified)
- **Preprocessing:** StandardScaler (zero mean, unit variance)
- **Random Seed:** 42 (reproducibility)

### 7.2 Performance Metrics

#### Overall Accuracy: **86.8%**

#### Confusion Matrix (Test Set, n=800):

|              | Predicted LOS | Predicted NLOS |
|--------------|---------------|----------------|
| **Actual LOS**  | 353 (88.2%)   | 47 (11.8%)     |
| **Actual NLOS** | 59 (14.8%)    | 341 (85.3%)    |

#### Classification Report:

| Metric | LOS | NLOS | Weighted Avg |
|--------|-----|------|--------------|
| **Precision** | 85.7% | 87.9% | 86.8% |
| **Recall** | 88.2% | 85.3% | 86.8% |
| **F1-Score** | 86.9% | 86.5% | 86.7% |

### 7.3 Feature Importance Analysis

**Logistic Regression Coefficients (Magnitude):**

| Rank | Feature | Coefficient | Interpretation |
|------|---------|-------------|----------------|
| 1 | **Max_Index** | -6.750 | Earlier max peak → LOS; Later → NLOS |
| 2 | **FP_INDEX_scaled** | +6.092 | Later FP detection → NLOS (signal penetration delay) |
| 3 | **first_bounce_delay_ns** | +1.658 | Longer delay → NLOS (complex reflections) |
| 4 | **fp_peak_amp** | -1.258 | Higher amplitude → LOS (less attenuation) |
| 5 | **multipath_count** | +1.251 | More peaks → NLOS (richer multipath) |
| 6 | **roi_energy** | -0.786 | Higher energy → LOS (weaker discriminator) |

**Top Discriminators:**
1. **Index-based features** (Max_Index, FP_INDEX_scaled) carry the most weight
2. **Temporal features** (bounce delay, multipath count) provide complementary information
3. **Energy/amplitude features** are less discriminative (confounded by distance)

### 7.4 Error Analysis

**False Positives (LOS predicted as NLOS): 47 cases (11.8%)**
- Likely borderline cases with unexpected multipath interference
- May include near-boundary LOS scenarios (e.g., corner reflections)

**False Negatives (NLOS predicted as LOS): 59 cases (14.8%)**
- NLOS signals with weak obstruction effects
- Open door scenario may resemble weak LOS conditions

**Balanced Performance:** Near-equal precision/recall for both classes indicates **no systematic bias**.

---

## 8. Key Insights & Conclusions

### 8.1 Major Findings

1. **Multipath Signature is Diagnostic:**
   - NLOS environments generate 27.8% more multipath components
   - First bounce timing shifts by 16.8% in NLOS conditions
   - These features provide robust classification signals

2. **Index Features Dominate:**
   - Hardware FP_INDEX and Max_Index are strongest predictors
   - Temporal positioning of signal peaks more reliable than amplitude/energy
   - Distance-normalized features may further improve performance

3. **Signal Stability Differs:**
   - LOS signals show consistent, repeatable waveforms
   - NLOS signals exhibit higher variance due to dynamic multipath

4. **Baseline Model is Effective:**
   - 86.8% accuracy with simple logistic regression
   - Balanced performance across classes
   - Room for improvement with advanced models (RF, XGBoost, Neural Networks)

### 8.2 Recommended Next Steps

#### Short-Term (Model Enhancement):
1. **Feature Engineering:**
   - Distance-normalized features (remove distance-dependency)
   - Statistical features (skewness, kurtosis of CIR)
   - Frequency domain features (FFT of CIR)

2. **Advanced Models:**
   - Random Forest / XGBoost (capture non-linear relationships)
   - 1D CNN (learn spatial patterns in CIR directly)
   - Ensemble methods (combine multiple weak learners)

3. **Cross-Validation:**
   - K-fold CV to assess model stability
   - Scenario-stratified splits to test generalization

#### Medium-Term (Deployment):
4. **Real-Time Classification:**
   - Optimize inference latency (<10ms target)
   - Feature extraction pipeline efficiency
   - Edge deployment on embedded systems

5. **Uncertainty Quantification:**
   - Prediction confidence scores
   - Anomaly detection for out-of-distribution inputs

6. **Integration with Positioning:**
   - NLOS-aware positioning algorithms
   - Weighted least squares with LOS/NLOS weights
   - Kalman filtering with dynamic measurement noise

#### Long-Term (Research):
7. **Dataset Expansion:**
   - More diverse environments (office, warehouse, outdoor-to-indoor)
   - Dynamic conditions (moving people, opening/closing doors)
   - Different UWB hardware (other transceivers)

8. **Multipath Exploitation:**
   - Use detected bounces for enhanced positioning (SLAM-like)
   - Ray-tracing validation against floor plans

---

## 9. Limitations & Considerations

### 9.1 Dataset Limitations

1. **Single Environment:** 
   - All data from one residential setting
   - Generalization to other environments uncertain

2. **Static Scenarios:**
   - No dynamic movement or time-varying channels
   - Real-world deployments face temporal variability

3. **Limited Distance Range:**
   - 1.56m - 4.4m (short-range focused)
   - Performance at longer ranges (>10m) unknown

### 9.2 Analysis Limitations

1. **Peak Detection Sensitivity:**
   - Threshold (5× noise floor) is heuristic
   - May miss weak multipath or detect false peaks

2. **ROI Selection:**
   - Fixed ROI (700-820) based on this dataset
   - Different hardware/settings may require adjustment

3. **Feature Independence:**
   - Some features are correlated (Max_Index ↔ FP_INDEX_scaled)
   - Multicollinearity may affect coefficient interpretation

---

## 10. Appendices

### A. Technical Specifications

**DW1000 Parameters:**
- **Chip:** DecaWave DW1000
- **Bandwidth:** 500 MHz
- **Center Frequency:** 6.5 GHz (Channel 5)
- **PRF:** 64 MHz
- **Preamble Code:** [Not specified in data]
- **CIR Samples:** 1,016 (15.65 ps resolution)

### B. Code Repository Structure

```
capstone/
├── dataset/              # Raw CSV files
├── eda/
│   ├── eda.ipynb        # This analysis notebook
│   ├── EDA_Report.md    # This report
│   ├── capture.py       # Data collection utilities
│   ├── los.py           # LOS-specific processing
│   └── nlos.py          # NLOS-specific processing
├── liquid/              # Model implementation
└── setup/               # Experiment setup scripts
```

### C. Reproducibility

All analysis is fully reproducible:
- **Random Seed:** 42 (fixed for train/test split)
- **Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Notebook:** `eda.ipynb` contains executable code
- **Environment:** Python 3.x with standard scientific stack

---

## References

1. Decawave DW1000 User Manual (v2.18)
2. IEEE 802.15.4a UWB Standard
3. Marano et al. (2010). "NLOS Identification and Mitigation for Localization Based on UWB Experimental Data"
4. Wymeersch et al. (2009). "A Machine Learning Approach to Ranging Error Mitigation for UWB Localization"

---

**Report End**

*For questions or additional analysis, please refer to the Jupyter Notebook: `eda.ipynb`*
