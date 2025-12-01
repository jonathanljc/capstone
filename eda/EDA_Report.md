# Exploratory Data Analysis Report
## UWB Localization: LOS/NLOS Classification

**Author:** Lim Jing Chuan Jonathan (2300923)  
**Date:** December 2, 2025  
**Project:** UWB Indoor Localization System  
**Notebook:** `eda.ipynb` (Reorganized with logical flow: Sections 1-10)

---

## Executive Summary

This report presents a comprehensive exploratory data analysis (EDA) of Ultra-Wideband (UWB) Channel Impulse Response (CIR) signals for Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) classification.

**Key Achievements:**
- ✅ **Dataset:** 8,000 balanced measurements (50% LOS, 50% NLOS) across 8 diverse scenarios in 3 environments
- ✅ **Baseline Performance:** 92.7% accuracy using logistic regression with 6 engineered features
- ✅ **Signal Features:** 30+ features engineered from CIR signals, with excellent discrimination (>23% difference)
- ✅ **Distance Range:** 1.56m - 8.34m across Home, Meeting Room, and Basement environments
- ✅ **Distance Error Analysis:** NLOS bias quantified (LOS: -0.087m, NLOS: -0.883m)
- ✅ **Notebook Organization:** Logical flow from basic features → multipath → signal analysis → hardware → distance

**What You'll Learn:**
1. How UWB signals differ between LOS and NLOS across diverse environments (Sections 1-3)
2. What features discriminate LOS from NLOS (Section 4: Feature Engineering in logical order)
3. **How signal and multipath characteristics reveal LOS/NLOS patterns** (Section 5: Comprehensive analysis) ⭐
4. **How distance error varies across scenarios** (Section 6: Scenario-specific bias) ⭐
5. **How CIR waveforms visualize propagation patterns** (Section 7: Multi-perspective visualization) ⭐
6. **Baseline classification performance and feature importance** (Section 8: 92.7% accuracy) ⭐

**Notebook Structure Reference (Reorganized for Logical Flow):**
- **Sections 1-3:** Configuration, Data Loading, Quality Checks
- **Section 4:** Feature Engineering (4.1 Basic → 4.2 Multipath → 4.3 Signal Analysis → 4.3b Hardware → 4.4 Distance)
- **Section 5:** Signal & Multipath Analysis (5.1 Box Plots → 5.2 Mean CIR → 5.3 Multipath Stats → 5.4 Statistical Validation → 5.5 Visual Comparison)
- **Section 6:** Distance Error Analysis by Scenario
- **Section 7:** CIR Waveform Visualization (7.1 Peak Detection → 7.2 Full Waveforms → 7.3 Stability Analysis)
- **Section 8:** Baseline Classification (Logistic Regression)
- **Section 9:** Dataset Summary & Export (9.1 Enhanced Dataset → 9.2 Merged Dataset)
- **Section 10:** Merged Dataset EDA & Validation

---

## PART 1: DATA UNDERSTANDING

### 1. Dataset Overview

**Hardware & Environment:**
- **Transceiver:** DecaWave DW1000 UWB chip
- **Signal:** Channel Impulse Response (CIR) - 1,016 samples per measurement
- **Time Resolution:** 15.65 ps per sample → Total duration = **15.9 ns**
- **Environments:** 3 diverse indoor locations (Home, Meeting Room, Basement)

**Scenarios (8 conditions across 3 environments):**

| Scenario | Type | Distance | Environment | Samples |
|----------|------|----------|-------------|---------|
| Living room | LOS | 2.0 m | Home - Clear line-of-sight | 1,000 |
| Living room corner | LOS | 4.3 m | Home - Clear line-of-sight | 1,000 |
| Meeting room | LOS | 4.63 m | SIT MR201 - Glass corner | 1,000 |
| Basement | LOS | 8.34 m | SIT E2B1 - Concrete corner | 1,000 |
| Open door | NLOS | 1.56 m | Home - Through open door | 1,000 |
| Meeting room | NLOS | 2.24 m | SIT MR201 - Table/laptop obstruction | 1,000 |
| Closed door | NLOS | 4.4 m | Home - Through closed door | 1,000 |
| Basement | NLOS | 7.67 m | SIT E2B1 - Thick concrete wall | 1,000 |

**Total:** 8,000 measurements (perfectly balanced: 4,000 LOS + 4,000 NLOS)

**Environmental Diversity:**
- **Home:** Living room scenarios with door obstructions (1.56m - 4.4m)
- **Meeting Room (SIT MR201):** Glass partitions and furniture interference (2.24m - 4.63m)
- **Basement (SIT E2B1):** Concrete walls with long-range propagation (7.67m - 8.34m)

**Data Quality:** ✅ No missing values, balanced classes, consistent CIR length, diverse distances (1.56m - 8.34m)

**Data Files Generated:**
- **`merged_cir.csv`** (Notebook Section 9.2) - Basic merge of 8 individual CSVs for fast loading
- **`merged_cir_enhanced.csv`** (Notebook Section 9.1) - Full dataset with all 30+ engineered features
- **Validation:** Section 10 confirms data integrity, scenario distribution, and environment balance

---

### 2. Physical Signal Characteristics

**What is CIR (Channel Impulse Response)?**
- The CIR captures how a UWB pulse spreads in time as it travels through an environment
- Each sample represents signal amplitude at a specific time offset (15.65 ps intervals)
- Index 0 = earliest arrival, Index 1015 = latest arrival (~16 ns later)

**Key Signal Features:**

| Characteristic | LOS Behavior | NLOS Behavior |
|---------------|--------------|---------------|
| **Peak sharpness** | Sharp, narrow peak | Broader, dispersed peak |
| **Signal decay** | Rapid fall-off after peak | Gradual decay with oscillations |
| **Multipath components** | Fewer reflections (13.6 avg) | More reflections (17.4 avg) → **+27.8%** |
| **First bounce delay** | Shorter (0.113 ns) | Longer (0.132 ns) → **+16.8%** |
| **Tail energy** | Concentrated (65.9%) | Dispersed (81.0%) → **+23.0%** |
| **Waveform stability** | Low variance | High variance |

**Physical Interpretation:**
- **LOS:** Direct path dominates, minimal multipath interference → clean, sharp signal
- **NLOS:** Signal penetrates/diffracts through obstacles → delayed, spread-out, multipath-rich signal

---

### 3. Feature Engineering Strategy

*Corresponds to Notebook Section 4 (4.1 → 4.2 → 4.3 → 4.3b → 4.4)*

We systematically extract features from raw CIR in 5 progressive stages:

#### 3.1 **Basic Features** (Notebook Section 4.1)

Foundation features for signal analysis:

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **ROI Energy** | Signal energy in Region of Interest (740-800) | `sum(CIR[740:800]^2)` |
| **FP_INDEX_scaled** | Hardware first-path detection (DW1000 chip) | `FP_INDEX / 64.0` |
| **Max_Index** | CIR index with maximum amplitude | `argmax(CIR)` |
| **True_Index** | Expected index from true distance | `round(d_true / c / TS_DW1000)` |
| **dist_error** | Hardware ranging error | `Distance - d_true` |

#### 3.2 **Multipath Features** (Notebook Section 4.2)

Extracted using peak detection algorithm (5× noise threshold):

| Feature | Description | LOS Mean | NLOS Mean | Difference |
|---------|-------------|----------|-----------|------------|
| **fp_peak_amp** | Amplitude at first path peak | Higher | Lower | More direct signal |
| **first_bounce_idx** | Index of first multipath bounce | Earlier | Later | Delayed reflections |
| **first_bounce_delay_ns** | Time to first bounce | 0.113 ns | 0.132 ns | **+16.8%** |
| **multipath_count** | Number of detected peaks | 13.6 | 17.4 | **+27.8%** |

#### 3.3 **Signal Analysis Features** (Notebook Section 4.3)

Advanced signal characteristics for temporal dynamics analysis:

| Feature | Physical Meaning | Formula | Discrimination |
|---------|-----------------|---------|----------------|
| **t_start** | Hardware first path detection | `FP_INDEX / 64` | Reference point |
| **t_peak** | Maximum peak index | `argmax(CIR)` | Reference point |
| **Rise_Time** | Signal rise (samples) | `t_peak - t_start` | ⭐⭐⭐⭐ |
| **Rise_Time_ns** | Signal rise (nanoseconds) | `Rise_Time × TS_DW1000 × 10^9` | ⭐⭐⭐⭐ |
| **RiseRatio** | Amplitude ratio | `CIR[t_start] / CIR[t_peak]` | ⭐⭐⭐⭐⭐ |
| **E_tail** | Tail energy ratio | `Σ(CIR[t_peak:end]^2) / Σ(CIR^2)` | ⭐⭐⭐⭐⭐ |
| **Peak_SNR** | Signal-to-noise ratio | `CIR[t_peak] / median_noise` | ⭐⭐⭐ |

**Key Statistics from Notebook Section 5.4:**

| Feature | LOS Mean | NLOS Mean | Difference |
|---------|----------|-----------|------------|
| **E_tail** | 65.9% | 81.0% | **+23.0%** |
| **RiseRatio** | Higher | Lower | Discriminative |
| **Peak_SNR** | Higher | Lower | Discriminative |

**Note:** The notebook correctly uses `FP_INDEX_scaled` as `t_start` to capture hardware-detected first arrival, ensuring proper rise time calculation.

#### 3.4 **Hardware-Derived Features** (Notebook Section 4.3b)

DW1000 chip hardware measurements:

| Feature | Description | Formula |
|---------|-------------|---------|
| **avg_fp_amplitude** | Average first path amplitude | `(FP_AMPL1 + FP_AMPL2 + FP_AMPL3) / 3` |
| **fp_amplitude_std** | First path stability | `std(FP_AMPL1, FP_AMPL2, FP_AMPL3)` |
| **signal_quality** | Preamble accumulation count | `RXPACC` |

#### 3.5 **Distance Components** (Notebook Section 4.4)

Distance-related features for error analysis:

| Component | Formula | Purpose |
|-----------|---------|---------|  
| **d_single_bounce** | `(FP_INDEX/64 × TS_DW1000) × c` | Hardware-based ranging |
| **d_error** | `d_single_bounce - d_true` | NLOS bias magnitude |
| **d_true** | Ground truth | Physical distance |

**Key Statistics from Notebook Section 4.4:**

| Metric | LOS | NLOS | Insight |
|--------|-----|------|---------|
| **d_error mean** | -0.087m | -0.883m | NLOS has 10× larger error |
| **d_error %** | -2.0% | -19.6% | Significant NLOS bias |

---

**Result:** With these 5 stages of feature engineering (20+ features total), baseline logistic regression achieves **92.7% accuracy** → Classification is highly feasible!

---

## PART 2: BASELINE CLASSIFICATION VALIDATION

*Corresponds to Notebook Section 8*

### 4. Logistic Regression Performance

**Model Setup:**
- Algorithm: Logistic Regression with StandardScaler
- Features: 6 engineered features (FP_INDEX_scaled, Max_Index, ROI_energy, fp_peak_amp, first_bounce_delay_ns, multipath_count)
- Train/Test Split: 80/20 (stratified)

**Results:**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **92.7%** |
| **LOS Precision/Recall** | 91.1% / 94.6% |
| **NLOS Precision/Recall** | 94.4% / 90.8% |
| **F1-Score (avg)** | 92.7% |

**Confusion Matrix (Test Set, n=1600):**

|              | Predicted LOS | Predicted NLOS |
|--------------|---------------|----------------|
| **Actual LOS**  | 757 (94.6%)   | 43 (5.4%)      |
| **Actual NLOS** | 74 (9.2%)     | 726 (90.8%)    |

**Feature Importance (Logistic Regression Coefficients):**

| Rank | Feature | Coefficient | Interpretation |
|------|---------|-------------|----------------|
| 1 | **Max_Index** | -6.198 | Earlier max peak → LOS |
| 2 | **fp_peak_amp** | -5.548 | Higher amplitude → LOS |
| 3 | **roi_energy** | -3.472 | Higher energy → LOS |
| 4 | **FP_INDEX_scaled** | +3.426 | Later first path → NLOS (penetration delay) |
| 5 | **multipath_count** | +3.030 | More peaks → NLOS |
| 6 | **first_bounce_delay_ns** | +1.044 | Longer bounce delay → NLOS |

**Key Insight:** Index-based temporal features (Max_Index, FP_INDEX_scaled) are the strongest discriminators!

---

## Conclusion

This exploratory data analysis has successfully characterized the key differences between LOS and NLOS UWB signals across 8 diverse scenarios in 3 environments:

**Key Findings:**
- ✅ **Dataset:** 8,000 balanced measurements across Home, Meeting Room, and Basement environments
- ✅ **Feature Engineering:** 30+ features extracted in logical order (basic → multipath → signal analysis → hardware → distance)
- ✅ **Discrimination:** Multipath count (+27.8%), tail energy (+23.0%), and rise characteristics show excellent separation
- ✅ **Baseline Classifier:** 92.7% accuracy with logistic regression validates classification feasibility
- ✅ **Distance Error:** NLOS bias quantified (LOS: -0.087m, NLOS: -0.883m) with scenario-specific patterns
- ✅ **Data Quality:** No missing values, balanced classes, comprehensive scenario coverage (1.56m - 8.34m)
- ✅ **Notebook Structure:** Reorganized for coherent analytical flow (Sections 1-10)

**Feature Importance Ranking:**
1. **Max_Index** (-6.198) - Peak location indicator (strongest)
2. **fp_peak_amp** (-5.548) - Signal amplitude
3. **roi_energy** (-3.472) - Signal energy
4. **FP_INDEX_scaled** (+3.426) - Hardware first-path detection
5. **multipath_count** (+3.030) - Reflection count
6. **first_bounce_delay_ns** (+1.044) - Multipath timing

**Physical Interpretation:**
- **LOS signals:** Sharp peaks, concentrated energy (65.9% tail energy), fewer reflections (13.6), shorter bounce delay (0.113ns)
- **NLOS signals:** Dispersed peaks, diffuse energy (81.0% tail energy), more reflections (17.4), longer bounce delay (0.132ns)

**Data Files:**
- `merged_cir.csv` - Basic merge of 8 CSVs for fast loading
- `merged_cir_enhanced.csv` - Full dataset with all engineered features (ready for modeling)

**Next Steps:**
- Advanced classification models (neural networks, ensemble methods)
- Distance regression for improved ranging accuracy
- Cross-environment validation and generalization testing

---

**Report Version:** 2.0 (Pure EDA - 8 Scenarios across 3 Environments)  
**Last Updated:** December 2, 2025  
**Author:** Lim Jing Chuan Jonathan (2300923)

---


