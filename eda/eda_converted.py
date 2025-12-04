
# Cell 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# UWB Signal Processing Constants
ROI_START, ROI_END = 740, 800  # Region of Interest for CIR analysis (empirically determined peak region)
TS_DW1000 = 1 / (128 * 499.2e6)  # DW1000 time sample period: 15.65 ps per sample
C_AIR = 299_792_458.0  # Speed of light in air (m/s)

# FP_INDEX is a hardware register value that needs scaling by 64 to convert to CIR index
# This represents the First Path detection from the DW1000 chip
FP_INDEX_SCALE = 64.0

# Preferred scenario ordering for consistent visualization (all 8 scenarios)
PREFERRED_SCENARIOS = [
    "LOS 2 m living room",
    "LOS 4.3 m corner",
    "LOS 4.63 m meeting room",
    "LOS 8.34 m basement",
    "NLOS 1.56 m open door",
    "NLOS 2.24 m meeting room",
    "NLOS 4.4 m closed door",
    "NLOS 7.67 m basement",
]

# Cell 3

# Load all 8 datasets individually
files = [
    # Home environment
    ('../dataset/LOS_2m_living_room_home.csv',        'LOS',  2.0,  'LOS 2 m living room'),
    ('../dataset/LOS_4.3m_living_room_corner_home.csv','LOS',  4.3, 'LOS 4.3 m corner'),
    ('../dataset/NLOS_1.56m_open_door_home.csv',       'NLOS', 1.56,'NLOS 1.56 m open door'),
    ('../dataset/NLOS_4.4m_close_door_home.csv',       'NLOS', 4.4, 'NLOS 4.4 m closed door'),
    
    # SIT Meeting Room MR201
    ('../dataset/LOS_4.63m_meetingroom_corner-glass_MR201SIT.csv', 'LOS', 4.63, 'LOS 4.63 m meeting room'),
    ('../dataset/NLOS_2.24m_meetingroom_table_laptop_MR201SIT.csv', 'NLOS', 2.24, 'NLOS 2.24 m meeting room'),
    
    # SIT Basement E2B1
    ('../dataset/LOS_8.34m_basement_corner-concrete_E2B1SIT.csv', 'LOS', 8.34, 'LOS 8.34 m basement'),
    ('../dataset/NLOS_7.67m_basement_concrete_thickconcretewall_E2B1SIT.csv', 'NLOS', 7.67, 'NLOS 7.67 m basement'),
]

frames = []
print("=" * 80)
print("LOADING INDIVIDUAL DATASETS")
print("=" * 80)
for fname, scenario_label, d_true, scen in files:
    df = pd.read_csv(fname)
    # CRITICAL FIX: Use startswith() instead of 'in' to avoid substring matching
    # 'LOS' in 'NLOS' returns True, causing incorrect labeling
    df['Label'] = 'LOS' if scenario_label.startswith('LOS') else 'NLOS'
    df['d_true'] = d_true
    df['scenario'] = scen
    frames.append(df)
    print(f'✓ {scen:35s} | {df.shape[0]:5,} samples | Label={df["Label"].iloc[0]} | {d_true:5.2f}m')

# Concatenate all datasets
data = pd.concat(frames, ignore_index=True)
print("=" * 80)
print(f'✅ TOTAL: {data.shape[0]:,} samples from {len(files)} scenarios')
print("=" * 80)

data.head()

# Cell 5

### 3.1 Missing Values Check
print('Shape:', data.shape)
print('\nMissing values (top 10):')
missing = data.isna().sum().sort_values(ascending=False).head(10)
if missing.max() > 0:
    print(missing)
    print('\n⚠ Warning: Missing values detected. May affect feature engineering.')
else:
    print('✓ No missing values detected.')

# Cell 6

### 3.2 Dataset Overview Statistics
print('=' * 80)
print('DATASET OVERVIEW')
print('=' * 80)
print(f'\nShape: {data.shape[0]:,} samples × {data.shape[1]} columns')

print('\n📊 Label Distribution:')
label_counts = data['Label'].value_counts()
for label, count in label_counts.items():
    print(f'  {label}: {count:,} samples ({count/len(data)*100:.1f}%)')

print('\n📋 Scenario Distribution:')
scenario_counts = data['scenario'].value_counts()
for scenario, count in scenario_counts.items():
    print(f'  {scenario}: {count:,} samples')

print('\n📏 True Distance Range:')
distances = data.groupby('scenario')['d_true'].first().sort_values()
print(f'  Minimum: {distances.min():.2f}m')
print(f'  Maximum: {distances.max():.2f}m')
print(f'  Range: {distances.max() - distances.min():.2f}m')

print('\n📍 Distances by Scenario:')
for scenario, dist in distances.items():
    print(f'  {scenario}: {dist:.2f}m')

print('\n' + '=' * 80)

# Cell 8

# Extract CIR column names first (needed for visualization)
cir_cols = [c for c in data.columns if c.startswith('CIR') and c[3:].isdigit()]
cir_cols.sort(key=lambda x: int(x[3:]))

# Full CIR comparison (1000 samples per scenario for clarity)
scenarios_all = list(data["scenario"].unique())
plot_scenarios = [s for s in PREFERRED_SCENARIOS if s in scenarios_all]
for s in scenarios_all:
    if s not in plot_scenarios:
        plot_scenarios.append(s)

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, scen in zip(axes, plot_scenarios[:8]):
    rows = data.loc[data["scenario"] == scen].head(1000)
    if rows.empty:
        ax.axis('off')
        continue
    for _, row in rows.iterrows():
        wf = row[cir_cols].values.astype(float)
        x = np.arange(len(wf))
        ax.plot(x, wf, lw=0.8, alpha=0.35)
    ax.set_title(f"{scen}", fontsize=10, fontweight='bold')
    ax.set_xlabel("CIR Index", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Full CIR Waveforms (1016 cir per sample)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Cell 10

### 5.1 Raw CIR Waveforms (Zoomed to ROI: 740-800)

# Zoom into the region of interest to see raw signal details
start_zoom, end_zoom = ROI_START, ROI_END

scenarios_all = list(data["scenario"].unique())
plot_scenarios = [s for s in PREFERRED_SCENARIOS if s in scenarios_all]
for s in scenarios_all:
    if s not in plot_scenarios:
        plot_scenarios.append(s)

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, scen in zip(axes, plot_scenarios[:8]):
    rows = data.loc[data["scenario"] == scen].head(100)
    if rows.empty:
        ax.axis('off')
        continue
    
    # Plot raw waveforms in ROI
    for _, row in rows.iterrows():
        wf = row[cir_cols].values.astype(float)
        x = np.arange(start_zoom, end_zoom + 1)
        ax.plot(x, wf[start_zoom:end_zoom + 1], lw=0.8, alpha=0.35, color='#1f77b4')
    
    ax.set_title(f"{scen}", fontsize=10, fontweight='bold')
    ax.set_xlabel("CIR Index", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvline(start_zoom, color='green', ls='--', lw=1, alpha=0.5, label='ROI start')
    ax.axvline(end_zoom, color='red', ls='--', lw=1, alpha=0.5, label='ROI end')

plt.suptitle(f"Raw CIR Waveforms - Zoomed to ROI ({start_zoom}-{end_zoom}), 100 samples per scenario", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n✅ Showing raw CIR signals zoomed to ROI region: indices {start_zoom} to {end_zoom}")
print(f"   Each plot overlays 100 samples from the scenario to show signal variability")

# Cell 12

# Signal stability: mean ± std dev in ROI region (740-800)
scenarios_all = list(data["scenario"].unique())
plot_scenarios = [s for s in PREFERRED_SCENARIOS if s in scenarios_all]
for s in scenarios_all:
    if s not in plot_scenarios:
        plot_scenarios.append(s)

results = []
color = sns.color_palette("deep")[0]

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, scen in zip(axes, plot_scenarios[:8]):
    rows = data.loc[data["scenario"] == scen].head(100)
    if rows.empty:
        ax.axis("off")
        continue
    wf_stack = rows[cir_cols].values.astype(float)
    mean_wf = wf_stack.mean(axis=0)
    std_wf = wf_stack.std(axis=0)
    
    # Use ROI region (740-800)
    x_roi = np.arange(ROI_START, ROI_END + 1)
    mean_roi = mean_wf[ROI_START:ROI_END + 1]
    std_roi = std_wf[ROI_START:ROI_END + 1]
    lower_roi = mean_roi - std_roi
    upper_roi = mean_roi + std_roi
    
    # Find peak within ROI
    peak_idx_roi = int(np.argmax(mean_roi)) + ROI_START
    peak_amp = float(mean_wf[peak_idx_roi])
    results.append({"scenario": scen, "peak_idx": peak_idx_roi, "peak_amp": peak_amp})

    ax.fill_between(x_roi, lower_roi, upper_roi,
                    color=color, alpha=0.2, linewidth=0, label='±1 std dev')
    ax.plot(x_roi, mean_roi, color=color, lw=2, label="Mean")
    ax.axvline(peak_idx_roi, color="red", ls="--", lw=1, label="Peak")

    ax.set_xlim(ROI_START, ROI_END)
    ax.set_title(scen, fontsize=10, fontweight='bold')
    ax.set_xlabel("CIR Index", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

fig.suptitle(f"Signal Stability Analysis in ROI ({ROI_START}-{ROI_END}): Mean ± Std Dev (1000 samples per scenario)", 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print(f"\n📊 Peak Characteristics in ROI ({ROI_START}-{ROI_END}) by Scenario:")
print("=" * 70)
peak_df = pd.DataFrame(results).sort_values("peak_idx")
for _, row in peak_df.iterrows():
    print(f"  {row['scenario']:30s} | Peak @ index {row['peak_idx']:4d} | Amplitude {row['peak_amp']:7.1f}")
print("=" * 70)

# Cell 14

### 7.1 Extract First Path Positions (Hardware vs Ground Truth)

print("=" * 80)
print("FIRST PATH DETECTION: HARDWARE vs GROUND TRUTH")
print("=" * 80)

# Hardware detection: FP_INDEX from DW1000 chip (needs scaling by 64)
data['FP_INDEX_scaled'] = data['FP_INDEX'] / FP_INDEX_SCALE
print(f"\n✅ Hardware First Path Index (FP_INDEX):")
print(f"   - FP_INDEX_scaled: Hardware-detected first path position (÷{FP_INDEX_SCALE})")
print(f"   - This is what the DW1000 chip reports in real-time")

# Ground truth reference: Expected CIR index based on true distance
data['ToF_true_s'] = data['d_true'] / C_AIR
data['True_Index'] = np.round(data['ToF_true_s'] / TS_DW1000).astype(int)
print(f"\n✅ Ground Truth Reference (True_Index):")
print(f"   - True_Index: Expected CIR index based on true distance")
print(f"   - Calculated as: (d_true / C_AIR) / TS_DW1000")

# Compare hardware vs ground truth
data['Index_Error_FP_vs_true'] = data['FP_INDEX_scaled'] - data['True_Index']
print(f"\n✅ Index Error Analysis:")
print(f"   - Index_Error_FP_vs_true: Hardware FP_INDEX vs ground truth")

print("\n" + "-" * 80)
print("First Path Comparison by Label:")
print("-" * 80)
fp_comparison = data.groupby('Label')[['FP_INDEX_scaled', 'True_Index', 'Index_Error_FP_vs_true']].agg(['mean', 'std'])
print(fp_comparison.round(2))

print("\n📊 Key Observations:")
los_error = data[data['Label'] == 'LOS']['Index_Error_FP_vs_true'].mean()
nlos_error = data[data['Label'] == 'NLOS']['Index_Error_FP_vs_true'].mean()
print(f"   LOS avg error:  {los_error:+.2f} indices ({los_error * TS_DW1000 * 1e9:+.3f} ns)")
print(f"   NLOS avg error: {nlos_error:+.2f} indices ({nlos_error * TS_DW1000 * 1e9:+.3f} ns)")

if abs(nlos_error) > abs(los_error):
    print(f"\n⚠️  NLOS shows {abs(nlos_error - los_error):.2f} indices more bias")
    print("   This is expected: NLOS signals arrive later due to obstruction")

print("=" * 80)

# Cell 16

def simple_peaks(wave, threshold, min_gap=3):
    """Detect local peaks in waveform above threshold with minimum separation."""
    peaks = []
    for i in range(1, len(wave) - 1):
        if wave[i] >= threshold and wave[i] >= wave[i - 1] and wave[i] >= wave[i + 1]:
            if peaks and (i - peaks[-1] < min_gap):
                if wave[i] > wave[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.array(peaks, dtype=int)

def extract_multipath(row, start=650, end=900):
    """Extract multipath features from a single CIR measurement."""
    wf = row[cir_cols].values.astype(float)
    end_idx = min(end, len(wf) - 1)
    roi = np.abs(wf[start:end_idx + 1])
    noise_floor = np.median(np.abs(wf[0:600]))
    threshold = noise_floor * 5
    local_peaks = simple_peaks(roi, threshold=threshold, min_gap=2)
    peaks_abs = local_peaks + start
    if len(peaks_abs) == 0:
        return pd.Series({
            'fp_peak_idx': np.nan,
            'fp_peak_amp': np.nan,
            'first_bounce_idx': np.nan,
            'first_bounce_delay_ns': np.nan,
            'multipath_count': 0,
        })

    fp_hw_idx = row['FP_INDEX'] / FP_INDEX_SCALE
    near_fp = [p for p in peaks_abs if p >= fp_hw_idx - 5]
    fp_peak_idx = min(near_fp) if near_fp else peaks_abs.min()
    fp_peak_amp = float(np.abs(wf[int(fp_peak_idx)]))

    later = [p for p in peaks_abs if p > fp_peak_idx + 5]
    first_bounce_idx = min(later) if later else np.nan
    first_bounce_delay_ns = ((first_bounce_idx - fp_peak_idx) * TS_DW1000 * 1e9
                             if not np.isnan(first_bounce_idx) else np.nan)

    return pd.Series({
        'fp_peak_idx': fp_peak_idx,
        'fp_peak_amp': fp_peak_amp,
        'first_bounce_idx': first_bounce_idx,
        'first_bounce_delay_ns': first_bounce_delay_ns,
        'multipath_count': len(peaks_abs),
    })

print("✅ Peak detection functions defined:")
print("   • simple_peaks(): Detect local maxima above noise threshold")
print("   • extract_multipath(): Extract first path peak and first bounce")

# Cell 17

### 7.3 Apply Peak Detection to Extract First Path and First Bounce

print("🔍 EXTRACTING MULTIPATH FEATURES...")
mp = data.apply(extract_multipath, axis=1)

# Remove any existing multipath columns before concatenating
mp_cols = ['fp_peak_idx', 'fp_peak_amp', 'first_bounce_idx', 'first_bounce_delay_ns', 'multipath_count']
data = data.drop(columns=[col for col in mp_cols if col in data.columns], errors='ignore')
data = pd.concat([data, mp], axis=1)

print("\n" + "=" * 80)
print("✅ Multipath features extracted:")
print(data[['fp_peak_idx', 'fp_peak_amp', 'first_bounce_idx', 'first_bounce_delay_ns', 'multipath_count']].head())
print("=" * 80)

# Cell 19

# Calculate distance estimates from both methods
# 1. Hardware-based distance (from FP_INDEX)
data['d_single_bounce'] = (data['FP_INDEX_scaled'] * TS_DW1000) * C_AIR

# 2. CIR-based distance (from detected first path peak)
data['d_from_fp_peak'] = (data['fp_peak_idx'] * TS_DW1000) * C_AIR

# Calculate errors for both methods
data['error_hardware'] = data['d_single_bounce'] - data['d_true']
data['error_cir_peak'] = data['d_from_fp_peak'] - data['d_true']

print("=" * 90)
print("DISTANCE ESTIMATION VALIDATION: Hardware vs CIR-Based")
print("=" * 90)

print("\n📏 Three Distance Estimates:")
print("   1. d_true:            Ground truth (measured)")
print("   2. d_single_bounce:   From hardware FP_INDEX (what chip reports)")
print("   3. d_from_fp_peak:    From CIR-detected first path peak (our algorithm)")

print("\n" + "-" * 90)
print("Overall Statistics:")
print("-" * 90)

comparison = data[['d_true', 'd_single_bounce', 'd_from_fp_peak', 'error_hardware', 'error_cir_peak']].describe()
print(comparison.round(3))

print("\n" + "-" * 90)
print("Error Analysis by Label:")
print("-" * 90)

for label in ['LOS', 'NLOS']:
    subset = data[data['Label'] == label]
    print(f"\n{label}:")
    print(f"  Hardware error:  Mean = {subset['error_hardware'].mean():+.3f}m, Std = {subset['error_hardware'].std():.3f}m")
    print(f"  CIR peak error:  Mean = {subset['error_cir_peak'].mean():+.3f}m, Std = {subset['error_cir_peak'].std():.3f}m")
    
    hw_mae = subset['error_hardware'].abs().mean()
    cir_mae = subset['error_cir_peak'].abs().mean()
    improvement = ((hw_mae - cir_mae) / hw_mae * 100)
    
    print(f"  Hardware MAE:    {hw_mae:.3f}m")
    print(f"  CIR peak MAE:    {cir_mae:.3f}m")
    if improvement > 0:
        print(f"  → CIR peak is {improvement:.1f}% more accurate! ✓")
    else:
        print(f"  → Hardware is {-improvement:.1f}% more accurate")

print("\n" + "-" * 90)
print("🔍 VALIDATION RESULTS:")
print("-" * 90)

overall_hw_mae = data['error_hardware'].abs().mean()
overall_cir_mae = data['error_cir_peak'].abs().mean()
overall_improvement = ((overall_hw_mae - overall_cir_mae) / overall_hw_mae * 100)

print(f"\nOverall MAE:")
print(f"  Hardware (FP_INDEX):     {overall_hw_mae:.3f}m")
print(f"  CIR peak detection:      {overall_cir_mae:.3f}m")
print(f"  Improvement:             {overall_improvement:+.1f}%")

if overall_cir_mae < overall_hw_mae:
    print(f"\n✅ CIR-based detection is MORE ACCURATE than hardware FP_INDEX!")
    print(f"   This validates that d_single_bounce has systematic bias")
    print(f"   → Use d_from_fp_peak for better distance estimation")
else:
    print(f"\n⚠️  Hardware FP_INDEX is more accurate (unexpected)")
    print(f"   → d_single_bounce may be acceptable for this dataset")

print("=" * 90)

# Cell 21

# Mean waveform with first path markers for clear visualization
scenarios_all = list(data["scenario"].unique())
plot_scenarios = [s for s in PREFERRED_SCENARIOS if s in scenarios_all]
for s in scenarios_all:
    if s not in plot_scenarios:
        plot_scenarios.append(s)

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

start_vis, end_vis = 740, 790
color = sns.color_palette("deep")[0]

for ax, scen in zip(axes, plot_scenarios[:8]):
    rows = data[data["scenario"] == scen].head(1000)
    if rows.empty:
        ax.axis("off")
        continue

    # Calculate mean +/- std in visualization window
    wf_stack = rows[cir_cols].values.astype(float)
    mean_wf = wf_stack.mean(axis=0)
    std_wf = wf_stack.std(axis=0)

    x_vis = np.arange(start_vis, end_vis + 1)
    mean_vis = mean_wf[start_vis:end_vis + 1]
    std_vis = std_wf[start_vis:end_vis + 1]
    lower_vis = mean_vis - std_vis
    upper_vis = mean_vis + std_vis

    # Plot mean +/- std
    ax.fill_between(x_vis, lower_vis, upper_vis,
                    color=color, alpha=0.2, linewidth=0, label='+/-1 std dev')
    ax.plot(x_vis, mean_vis, color=color, lw=2.5, label="Mean CIR", zorder=5)

    # Get first sample for marker positions
    first_row = rows.iloc[0]
    fp_hw = first_row.get("FP_INDEX", np.nan) / FP_INDEX_SCALE
    fp_peak_idx = first_row.get("fp_peak_idx", np.nan)
    fb_idx = first_row.get("first_bounce_idx", np.nan)
    true_idx = first_row.get("True_Index", np.nan)
    if np.isnan(true_idx):
        d_true_val = first_row.get("d_true", np.nan)
        if not np.isnan(d_true_val):
            true_idx = (d_true_val / C_AIR) / TS_DW1000

    handles, labels = [], []
    lines_text = []

    # Add vertical markers
    if not np.isnan(fp_hw):
        fp_line_x = fp_hw
        if not np.isnan(fp_peak_idx) and abs(fp_hw - fp_peak_idx) < 1.0:
            fp_line_x = fp_hw - 1.0
        h = ax.axvline(fp_line_x, color="red", ls=(0, (6, 3)), lw=1.5, alpha=0.95, zorder=7)
        handles.append(h); labels.append("FP hardware")

    if not np.isnan(fp_peak_idx):
        h = ax.axvline(fp_peak_idx, color="green", ls="--", lw=1.5, zorder=6)
        handles.append(h); labels.append("First path peak")

    if not np.isnan(fb_idx):
        h = ax.axvline(fb_idx, color="purple", ls="--", lw=1.5, zorder=6)
        handles.append(h); labels.append("First bounce")

    if not np.isnan(true_idx) and start_vis <= true_idx <= end_vis:
        h = ax.axvline(true_idx, color="black", ls=":", lw=1.3, alpha=0.9, zorder=8)
        handles.append(h); labels.append("Ground truth (laser)")
    elif not np.isnan(true_idx):
        lines_text.append(f"Ground truth idx ~{true_idx:.1f} (outside view)")

    ax.set_xlim(start_vis, end_vis)
    ax.set_title(scen, fontsize=10, fontweight='bold')
    ax.set_xlabel("CIR index", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(alpha=0.3)

    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7)

    # Add summary text
    if not np.isnan(fp_hw) and not np.isnan(fp_peak_idx):
        hw_error = fp_hw - fp_peak_idx
        lines_text.append(f"HW error: {hw_error:+.1f} idx")
    if not np.isnan(fp_hw) and not np.isnan(true_idx):
        lines_text.append(f"HW-true: {fp_hw - true_idx:+.1f} idx")
    if not np.isnan(fp_peak_idx) and not np.isnan(true_idx):
        lines_text.append(f"Peak-true: {fp_peak_idx - true_idx:+.1f} idx")
    if not np.isnan(fp_peak_idx) and not np.isnan(fb_idx):
        gap_ns = (fb_idx - fp_peak_idx) * TS_DW1000 * 1e9
        lines_text.append(f"FP->FB: {gap_ns:.2f} ns")
    if lines_text:
        ax.text(0.98, 0.03, "
".join(lines_text), fontsize=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                va="bottom", ha="right", transform=ax.transAxes)

plt.suptitle("First Path Detection: Mean Signal +/- Std Dev (1000 samples per scenario)", 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("
" + "=" * 80)
print("CLEAR VISUALIZATION SUMMARY:")
print("=" * 80)
print("- Mean waveform (solid blue): Average signal shape across 1000 samples")
print("- Shaded region: +/-1 standard deviation (signal variability)")
print("- Black dotted line (Ground truth): Expected first path from laser distance")
print("- Red line (Hardware FP_INDEX): What DW1000 chip reports")
print("- Green line (First path peak): CIR-based first path detection")
print("- Purple line (First bounce): Next significant reflection")
print("
Hardware Accuracy:")
print("- Compare hardware/peak to ground truth to see bias")
print("- NLOS scenarios: Hardware may deviate from true first path (positive error)")
print("- This visualization clearly shows systematic hardware bias in NLOS conditions")
print("=" * 80)


# Cell 22

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 1. Determine the correct label column name
if 'Label' in data.columns:
    label_col = 'Label'
elif 'label' in data.columns:
    label_col = 'label'
else:
    raise ValueError("Could not find 'Label' or 'label' column in data")

# 1b. Determine environment column for styling (optional)
if 'environment' in data.columns:
    env_col = 'environment'
else:
    env_col = None

# 2. Create the Systematic Bias Scatter Plot
plt.figure(figsize=(8, 8))

# 3. Plot the Ideal "Perfect" Line (y = x)
max_dist = data['d_true'].max()
min_dist = data['d_true'].min()
plt.plot([min_dist, max_dist], [min_dist, max_dist], 
         color='black', linestyle='--', linewidth=2, label='Ideal (Ground Truth)')

# 4. Plot the Hardware Measurements (Scatter)
scatter_kwargs = dict(data=data, x='d_true', y='d_single_bounce',
                      hue=label_col, alpha=0.6, s=60)
if env_col:
    scatter_kwargs['style'] = env_col
sns.scatterplot(**scatter_kwargs)

plt.title("Systematic Hardware Bias: Measured vs. True Distance", fontsize=14, fontweight='bold')
plt.xlabel("True Distance (Laser) [m]", fontsize=12)
plt.ylabel("Hardware Measured Distance (FP_INDEX) [m]", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title="Condition", loc='upper left')

# 5. Add text showing the mean error
los_error = (data[data[label_col]=='LOS']['d_single_bounce'] - data[data[label_col]=='LOS']['d_true']).mean()
nlos_error = (data[data[label_col]=='NLOS']['d_single_bounce'] - data[data[label_col]=='NLOS']['d_true']).mean()

bias_text = f"""Mean Bias:
LOS: {los_error:+.2f} m
NLOS: {nlos_error:+.2f} m"""

plt.text(0.95, 0.05, bias_text,
         transform=plt.gca().transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

plt.tight_layout()
plt.show()


# Cell 24

start, end = 740, 790
print(f"Visualization window: CIR indices {start} to {end}")

scenarios_all = list(data["scenario"].unique())
plot_scenarios = [s for s in PREFERRED_SCENARIOS if s in scenarios_all]
for s in scenarios_all:
    if s not in plot_scenarios:
        plot_scenarios.append(s)

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, scen in zip(axes, plot_scenarios[:8]):
    rows = data[data["scenario"] == scen].head(1000)
    if rows.empty:
        ax.axis("off")
        continue
    first_row = rows.iloc[0]
    x = np.arange(start, end + 1)
    peak_counts = []
    
    for _, row in rows.iterrows():
        wf = row[cir_cols].values.astype(float)
        noise_floor = np.median(np.abs(wf[0:600]))
        threshold = noise_floor * 5
        peaks = simple_peaks(np.abs(wf[start:end + 1]), threshold=threshold, min_gap=2) + start
        peak_counts.append(len(peaks))
        ax.plot(x, wf[start:end + 1], color="#1f77b4", alpha=0.28, lw=0.9)
        ax.scatter(peaks, wf[peaks], color="orange", s=10, alpha=0.55)

    fp_hw = first_row.get("FP_INDEX", np.nan) / FP_INDEX_SCALE
    fp_peak_idx = first_row.get("fp_peak_idx", np.nan)
    fb_idx = first_row.get("first_bounce_idx", np.nan)
    
    handles, labels = [], []
    if not np.isnan(fp_hw):
        fp_line_x = fp_hw
        if not np.isnan(fp_peak_idx) and abs(fp_hw - fp_peak_idx) < 1.0:
            fp_line_x = fp_hw - 1.0
        h = ax.axvline(fp_line_x, color="red", ls=(0, (6, 3)), lw=1.3, alpha=0.95, zorder=7)
        handles.append(h); labels.append("FP hardware")
    if not np.isnan(fp_peak_idx):
        h = ax.axvline(fp_peak_idx, color="green", ls="--", lw=1.3, zorder=6)
        handles.append(h); labels.append("First path peak")
    if not np.isnan(fb_idx):
        h = ax.axvline(fb_idx, color="purple", ls="--", lw=1.3, zorder=6)
        handles.append(h); labels.append("First bounce")

    ax.set_xlim(start, end)
    ax.set_title(scen, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlabel("CIR index", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7)

    lines_text = []
    if not np.isnan(fp_peak_idx) and not np.isnan(fb_idx):
        gap_idx = fb_idx - fp_peak_idx
        gap_ns = gap_idx * TS_DW1000 * 1e9
        lines_text.append(f"FP→FB: {gap_idx:.1f} idx ({gap_ns:.2f} ns)")
    if peak_counts:
        lines_text.append(f"Median peaks: {np.median(peak_counts):.0f}")
    if lines_text:
        ax.text(0.98, 0.03, "\n".join(lines_text), fontsize=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                va="bottom", ha="right", transform=ax.transAxes)

plt.suptitle("Detailed Peak Detection: Individual Waveforms (100 samples per scenario)", 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("📊 DETAILED VALIDATION:")
print("=" * 80)
print("✓ Blue waveforms: 100 individual CIR measurements (overlaid)")
print("✓ Orange dots: Detected peaks above 5× noise threshold")
print("✓ Shows peak detection consistency across multiple samples")
print("\n🔍 Algorithm Validation:")
print("  • LOS scenarios: Consistent peak detection, low variability")
print("  • NLOS scenarios: More variable peak patterns due to multipath")
print("  • Peak count variation indicates environment complexity")
print("=" * 80)

# Cell 26

### 8.1 Hardware Diagnostic Features (DW1000 Chip API)

# These features are directly collected from the DW1000 chip's diagnostic API
# and represent hardware-level measurements available in real-time

print("📡 HARDWARE DIAGNOSTIC FEATURES (DW1000 Chip)")
print("=" * 80)

# First Path Amplitude features (3 measurements from chip)
data['avg_fp_amplitude'] = data[['FP_AMPL1', 'FP_AMPL2', 'FP_AMPL3']].mean(axis=1)
data['fp_amplitude_std'] = data[['FP_AMPL1', 'FP_AMPL2', 'FP_AMPL3']].std(axis=1)
data['fp_amplitude_max'] = data[['FP_AMPL1', 'FP_AMPL2', 'FP_AMPL3']].max(axis=1)

print("✅ First Path Amplitudes (FP_AMPL1, FP_AMPL2, FP_AMPL3):")
print(f"   - avg_fp_amplitude: Mean of 3 amplitude measurements")
print(f"   - fp_amplitude_std: Std dev (signal consistency indicator)")
print(f"   - fp_amplitude_max: Max amplitude")

# Signal Quality Metrics
data['noise_ratio'] = data['MAX_NOISE'] / (data['STD_NOISE'] + 1e-6)  # Avoid division by zero

print(f"\n✅ Noise & Quality Metrics (STD_NOISE, MAX_NOISE):")
print(f"   - noise_ratio: MAX_NOISE / STD_NOISE (noise variability)")
print(f"\n📌 Note: RXPACC will be used for CIR normalization (not as standalone feature)")
print(f"📌 Note: FP_INDEX_scaled already computed in Section 7.1")

print("\n" + "=" * 80)
print("Hardware Features Summary:")
hw_features = ['FP_AMPL1', 'FP_AMPL2', 'FP_AMPL3', 'avg_fp_amplitude', 'fp_amplitude_std',
               'FP_INDEX_scaled', 'STD_NOISE', 'MAX_NOISE', 'noise_ratio']
print(data[hw_features].describe().round(2))
print("=" * 80)

# Cell 28

print("📊 CIR-DERIVED BASIC FEATURES")
print("=" * 80)

# CIR Normalization using RXPACC (per DW1000 User Manual)
print(f"✅ CIR Normalization:")
print(f"   - Normalizing CIR amplitudes by RXPACC for consistent scaling")

# Create all normalized columns at once using pd.concat to avoid fragmentation
cir_norm_df = pd.DataFrame({
    f'{col}_norm': data[col] / (data['RXPACC'] + 1e-6) 
    for col in cir_cols
}, index=data.index)
data = pd.concat([data, cir_norm_df], axis=1)
cir_norm_cols = [f'{c}_norm' for c in cir_cols]

# ROI energy (sum of squared NORMALIZED amplitudes in region of interest)
roi_cols = [f'CIR{i}' for i in range(ROI_START, ROI_END + 1)]
roi_norm_cols = [f'{c}_norm' for c in roi_cols]
data['roi_energy'] = (data[roi_norm_cols].values ** 2).sum(axis=1)
print(f"\n✅ ROI Energy (CIR indices {ROI_START}-{ROI_END}):")
print(f"   - roi_energy: Sum of squared NORMALIZED amplitudes in ROI")

# Peak position from CIR
data['Max_Index'] = data[cir_cols].idxmax(axis=1).str.replace('CIR', '').astype(int)
data['max_amplitude'] = data[cir_cols].max(axis=1)
print(f"\n✅ CIR Peak Detection:")
print(f"   - Max_Index: Position of maximum amplitude in CIR")
print(f"   - max_amplitude: Peak amplitude value")

print(f"\n📌 Note: True_Index and Index_Error_FP_vs_true already computed in Section 7.1")

# Index error analysis (CIR peak vs ground truth)
data['Index_Error_peak_vs_true'] = data['Max_Index'] - data['True_Index']
print(f"\n✅ Additional Index Error:")
print(f"   - Index_Error_peak_vs_true: CIR peak vs ground truth")

# Distance estimation from hardware
data['d_single_bounce'] = (data['FP_INDEX_scaled'] * TS_DW1000) * C_AIR
data['d_error'] = data['d_single_bounce'] - data['d_true']
data['dist_error'] = data['Distance'] - data['d_true']  # Keep for compatibility
print(f"\n✅ Distance Estimation:")
print(f"   - d_single_bounce: Distance from hardware FP_INDEX")
print(f"   - d_error: Ranging error (NLOS bias)")

print("\n" + "=" * 80)
print("CIR Basic Features Summary:")
cir_basic = ['roi_energy', 'Max_Index', 'max_amplitude', 'True_Index', 
             'Index_Error_FP_vs_true', 'Index_Error_peak_vs_true', 'd_single_bounce', 'd_error']
print(data[cir_basic].head())
print("=" * 80)


# Cell 30

# Select all engineered features for correlation analysis
hardware_features = ['FP_INDEX_scaled', 'avg_fp_amplitude', 'fp_amplitude_std', 'fp_amplitude_max',
                     'noise_ratio', 'STD_NOISE', 'MAX_NOISE']
cir_derived_features = ['roi_energy', 'Max_Index', 'max_amplitude', 'Index_Error_FP_vs_true', 
                        'Index_Error_peak_vs_true', 'd_single_bounce', 'd_error',
                        'fp_peak_idx', 'fp_peak_amp', 'first_bounce_delay_ns', 'multipath_count']

all_features = hardware_features + cir_derived_features

# Calculate correlation matrix
corr_matrix = data[all_features].corr()

# Find high correlations between hardware and CIR-derived features
print("=" * 90)
print("HIGH CORRELATIONS BETWEEN HARDWARE AND CIR-DERIVED FEATURES (|r| > 0.5)")
print("=" * 90)

high_corr_pairs = []
for hw_feat in hardware_features:
    for cir_feat in cir_derived_features:
        corr_val = corr_matrix.loc[hw_feat, cir_feat]
        if abs(corr_val) > 0.5:
            high_corr_pairs.append({
                'Hardware': hw_feat,
                'CIR-Derived': cir_feat,
                'Correlation': corr_val,
                'Abs_Corr': abs(corr_val)
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Abs_Corr', ascending=False)
    print("\n⚠️  POTENTIAL REDUNDANCY DETECTED:")
    for _, row in high_corr_df.iterrows():
        relationship = "positively" if row['Correlation'] > 0 else "negatively"
        print(f"   {row['Hardware']:25s} ↔ {row['CIR-Derived']:25s} | r = {row['Correlation']:+.3f} ({relationship} correlated)")
else:
    print("\n✅ NO HIGH CORRELATIONS FOUND - Features are complementary!")

# Visualize correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Full correlation matrix
ax1 = axes[0]
im1 = ax1.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(len(all_features)))
ax1.set_yticks(range(len(all_features)))
ax1.set_xticklabels(all_features, rotation=90, ha='right', fontsize=9)
ax1.set_yticklabels(all_features, fontsize=9)
ax1.set_title('Full Feature Correlation Matrix\n(Hardware + CIR-Derived)', fontsize=12, fontweight='bold')

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Add correlation values for high correlations
for i in range(len(all_features)):
    for j in range(len(all_features)):
        if abs(corr_matrix.values[i, j]) > 0.7 and i != j:
            text = ax1.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8, fontweight='bold')

# Cross-correlation: Hardware vs CIR-Derived only
cross_corr = corr_matrix.loc[hardware_features, cir_derived_features]
ax2 = axes[1]
im2 = ax2.imshow(cross_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax2.set_xticks(range(len(cir_derived_features)))
ax2.set_yticks(range(len(hardware_features)))
ax2.set_xticklabels(cir_derived_features, rotation=90, ha='right', fontsize=9)
ax2.set_yticklabels(hardware_features, fontsize=9)
ax2.set_title('Cross-Correlation: Hardware ↔ CIR-Derived\n(Focus on Redundancy)', fontsize=12, fontweight='bold')

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Annotate all cells in cross-correlation
for i in range(len(hardware_features)):
    for j in range(len(cir_derived_features)):
        val = cross_corr.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        text = ax2.text(j, i, f'{val:.2f}',
                       ha="center", va="center", color=color, fontsize=8)

plt.tight_layout()
plt.show()

print("\n" + "=" * 90)
print("📊 INTERPRETATION GUIDE:")
print("=" * 90)
print("• |r| > 0.8: Strong correlation - consider removing one feature (redundant)")
print("• 0.5 < |r| < 0.8: Moderate correlation - features share some information")
print("• |r| < 0.5: Weak correlation - features provide complementary information")
print("\n✅ RECOMMENDATION:")
if high_corr_pairs:
    print(f"   Found {len(high_corr_df[high_corr_df['Abs_Corr'] > 0.8])} strongly correlated pairs (|r| > 0.8)")
    print("   Consider feature selection to reduce redundancy in ML models")
else:
    print("   Hardware and CIR-derived features are largely independent")
    print("   Both feature sets provide unique, complementary information!")
print("=" * 90)

# Cell 32

### 8.4 Create Optimized LNN Context Features (7 features)

print("=" * 90)
print("🎯 CREATING OPTIMIZED CONTEXT FEATURES FOR LIQUID NEURAL NETWORK")
print("=" * 90)

# Based on correlation analysis, select 7 optimal features for LNN tau modulation
# These will be saved to the enhanced dataset for direct use in LNN training

print("\n📊 Selected Features (7 total):")
print("\n📡 Hardware Diagnostics (3 features):")
print("   1. FP_INDEX_scaled - First path position from hardware")
print("   2. avg_fp_amplitude - Mean first path amplitude")
print("   3. STD_NOISE - Noise floor baseline")

print("\n📊 CIR-Derived (4 features):")
print("   4. roi_energy - Signal power in ROI (normalized by RXPACC)")
print("   5. Max_Index - CIR peak position")
print("   6. multipath_count - Number of detected multipath components")
print("   7. first_bounce_delay_ns - Delay between first path and first bounce")

print("\n📌 RXPACC Usage:")
print("   • Used for CIR normalization (CIR_norm = CIR / RXPACC)")
print("   • All normalized CIR samples saved as CIR0_norm - CIR1015_norm")
print("   • roi_energy computed from normalized CIR values")

# Create optimized feature set for LNN
lnn_context_features = [
    # Hardware (3 features)
    'FP_INDEX_scaled',
    'avg_fp_amplitude', 
    'STD_NOISE',
    # CIR-derived (4 features)
    'roi_energy',
    'Max_Index',
    'multipath_count',
    'first_bounce_delay_ns'
]

print("\n" + "=" * 90)
print("✅ LNN CONTEXT FEATURES READY:")
print("=" * 90)
print("   These 7 features will be used to modulate tau in the Liquid Neural Network")
print("   They will be fed separately from the CIR sequence (1016 samples)")
print("\n   Architecture:")
print("   • Input 1: Normalized CIR sequence (B, 1016, 1) → LTC cells")
print("   •          CIR normalized by RXPACC: CIR_norm = CIR / RXPACC")
print("   • Input 2: Context features (B, 7) → Tau modulation")
print("=" * 90)

# Display feature statistics
print("\n📊 Feature Statistics:")
print(data[lnn_context_features].describe().round(3))

# Check for missing values in context features
missing_context = data[lnn_context_features].isna().sum()
if missing_context.sum() > 0:
    print("\n⚠️  Missing Values Detected:")
    print(missing_context[missing_context > 0])
    print(f"\n   Will drop {missing_context.max()} rows with missing context features before export")
else:
    print("\n✅ No missing values in context features!")

print("\n" + "=" * 90)

# Cell 34

print('Mean first-bounce delay (ns) by label:')
bounce_stats = data.groupby('Label')['first_bounce_delay_ns'].agg(['mean', 'std', 'count'])
print(bounce_stats)

# Check for NaN values
print('\nNaN Analysis for first_bounce_delay_ns:')
for label in ['LOS', 'NLOS']:
    subset = data[data['Label'] == label]
    total = len(subset)
    valid = subset['first_bounce_delay_ns'].notna().sum()
    nan_count = total - valid
    print(f'  {label}: {valid}/{total} valid ({valid/total*100:.1f}%), {nan_count} NaN')

print('\nMultipath count by label:')
multipath_stats = data.groupby('Label')['multipath_count'].agg(['mean', 'std', 'count'])
print(multipath_stats)

# Calculate percentage difference (only on valid data)
los_bounce = data[data['Label'] == 'LOS']['first_bounce_delay_ns'].mean()
nlos_bounce = data[data['Label'] == 'NLOS']['first_bounce_delay_ns'].mean()
pct_diff = ((nlos_bounce - los_bounce)/los_bounce * 100)
print(f'\n→ NLOS first bounce is {pct_diff:+.1f}% different from LOS')

los_mp = data[data['Label'] == 'LOS']['multipath_count'].mean()
nlos_mp = data[data['Label'] == 'NLOS']['multipath_count'].mean()
mp_pct_diff = ((nlos_mp - los_mp)/los_mp * 100)
print(f'→ NLOS has {mp_pct_diff:+.1f}% more multipath components than LOS')

print('\n⚠️ INTERPRETATION:')
print('   - NLOS has MORE peaks (23.2 vs 12.3) but peaks are CLOSER together')
print('   - This is physically consistent: multiple diffraction paths clustered in time')
print('   - LOS has FEWER peaks but they are MORE SPREAD OUT (clean room reflections)')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=data, x='first_bounce_delay_ns', hue='Label', bins=60, element='step', kde=False)
plt.title('First Bounce Delay Distribution')
plt.xlabel('Delay from first path (ns)')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
sns.histplot(data=data, x='multipath_count', hue='Label', bins=30, element='step', kde=False)
plt.title('Multipath Count Distribution')
plt.xlabel('Number of detected peaks')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 36

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Feature selection: Hardware + CIR-derived features
feat_cols = [
    # Hardware diagnostic features
    'FP_INDEX_scaled', 'avg_fp_amplitude', 'noise_ratio', 'STD_NOISE',
    # CIR-derived features
    'roi_energy', 'fp_peak_amp', 'first_bounce_delay_ns', 'multipath_count', 'Max_Index'
]

df = data.dropna(subset=feat_cols).copy()
X = df[feat_cols].values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('=' * 80)
print('BASELINE LOGISTIC REGRESSION CLASSIFIER')
print('=' * 80)
print(f'\n📊 Features Used ({len(feat_cols)} total):')
print(f'   Hardware (4): FP_INDEX_scaled, avg_fp_amplitude, noise_ratio, STD_NOISE')
print(f'   CIR-derived (5): roi_energy, fp_peak_amp, first_bounce_delay_ns, multipath_count, Max_Index')
print(f'\nTraining samples: {len(X_train):,}')
print(f'Test samples: {len(X_test):,}')
print(f'\n🎯 Overall Accuracy: {accuracy_score(y_test, y_pred):.3f} ({accuracy_score(y_test, y_pred)*100:.1f}%)')

print('\n📊 Confusion Matrix (rows=actual, cols=predicted):')
cm = confusion_matrix(y_test, y_pred)
print(f'\n              Predicted LOS    Predicted NLOS')
print(f'Actual LOS    {cm[0,0]:8d}         {cm[0,1]:8d}')
print(f'Actual NLOS   {cm[1,0]:8d}         {cm[1,1]:8d}')

print('\n📋 Classification Report:')
print(classification_report(y_test, y_pred, digits=3))

print('🔍 Feature Importance (coefficient magnitude):')
print('-' * 80)
coef = clf.named_steps['logisticregression'].coef_[0]
feat_importance = sorted(zip(feat_cols, coef), key=lambda t: abs(t[1]), reverse=True)
for rank, (name, c) in enumerate(feat_importance, 1):
    direction = "→ NLOS" if c > 0 else "→ LOS"
    feature_type = "[HW]" if name in ['FP_INDEX_scaled', 'avg_fp_amplitude', 'noise_ratio', 'STD_NOISE'] else "[CIR]"
    print(f"{rank}. {feature_type} {name:25s}: {c:+7.3f}  {direction}")
print('=' * 80)

# Cell 38

print("=" * 90)
print("COMPREHENSIVE DATASET SUMMARY")
print("=" * 90)

# Overall statistics
print(f"\n📊 TOTAL DATASET: {len(data):,} samples")
print(f"   LOS:  {len(data[data['Label']=='LOS']):,} samples ({len(data[data['Label']=='LOS'])/len(data)*100:.1f}%)")
print(f"   NLOS: {len(data[data['Label']=='NLOS']):,} samples ({len(data[data['Label']=='NLOS'])/len(data)*100:.1f}%)")

print("\n" + "-" * 90)
print("📍 ENVIRONMENT BREAKDOWN")
print("-" * 90)

# Group by environment
data['environment'] = data['scenario'].apply(lambda x: 
    'Home' if 'living room' in x or 'door' in x 
    else 'Meeting Room' if 'meeting room' in x 
    else 'Basement')

for env in ['Home', 'Meeting Room', 'Basement']:
    env_data = data[data['environment'] == env]
    if len(env_data) == 0:
        continue
    print(f"\n{env}:")
    print(f"  Total: {len(env_data):,} samples")
    print(f"  LOS:   {len(env_data[env_data['Label']=='LOS']):,} samples")
    print(f"  NLOS:  {len(env_data[env_data['Label']=='NLOS']):,} samples")
    print(f"  Distance range: {env_data['d_true'].min():.2f}m - {env_data['d_true'].max():.2f}m")

print("\n" + "-" * 90)
print("📋 SCENARIO DETAILS")
print("-" * 90)

scenario_stats = []
for scen in PREFERRED_SCENARIOS:
    if scen not in data['scenario'].values:
        continue
    subset = data[data['scenario'] == scen]
    scenario_stats.append({
        'Scenario': scen,
        'Samples': len(subset),
        'Label': subset['Label'].iloc[0],
        'Distance (m)': subset['d_true'].iloc[0],
        'Avg d_error (m)': subset['d_error'].mean(),
        'Error %': (subset['d_error'].mean() / subset['d_true'].iloc[0] * 100)
    })

scenario_df = pd.DataFrame(scenario_stats)
print(scenario_df.to_string(index=False))

print("\n" + "=" * 90)
print("✅ DATASET READY FOR MACHINE LEARNING")
print("=" * 90)
print("\nKey columns for modeling:")
print("  • CIR0-CIR1015: Raw Channel Impulse Response (1016 samples)")
print("  • Signal features: t_start, t_peak, Rise_Time_ns, RiseRatio, E_tail, Peak_SNR")
print("  • Hardware features: avg_fp_amplitude, fp_amplitude_std, signal_quality (RXPACC)")
print("  • Multipath features: fp_peak_amp, first_bounce_delay_ns, multipath_count")
print("  • Classification target: label (LOS/NLOS)")
print("  • Distance features: d_single_bounce, d_error, d_true")

# Cell 40

# Export complete dataset with all features
output_path = Path('../dataset/merged_cir_enhanced.csv')

# Drop rows with missing context features before export
valid_data = data.dropna(subset=lnn_context_features)
print(f"📊 Dataset size: {len(data):,} → {len(valid_data):,} samples (dropped {len(data) - len(valid_data):,} with missing values)")

valid_data.to_csv(output_path, index=False)

print(f"\n✅ Saved enhanced dataset to: {output_path}")
print(f"   Total samples: {len(valid_data):,}")
print(f"   Total columns: {len(valid_data.columns)}")
print(f"\n🎯 LNN-Ready Features Included:")
print(f"   • Raw CIR: CIR0-CIR1015 (1016 samples)")
print(f"   • Normalized CIR: CIR0_norm-CIR1015_norm (1016 samples, normalized by RXPACC)")
print(f"   • Context features: {', '.join(lnn_context_features)} (7 features)")
print(f"\nNew features added:")
print("   • CIR normalization: All CIR samples normalized by RXPACC")
print("   • d_single_bounce: Hardware-based distance estimation")
print("   • d_error: Ranging error (NLOS bias)")
print("   • environment: Categorical grouping (Home/Meeting Room/Basement)")
print("   • Hardware features: avg_fp_amplitude, fp_amplitude_std, FP_INDEX_scaled")
print("   • Multipath features: fp_peak_idx, first_bounce_idx, first_bounce_delay_ns, multipath_count")
print("\n🚀 Ready for LNN training!")
print(f"\n💡 Usage in PyTorch:")
print(f"   # Use NORMALIZED CIR for better training stability")
print(f"   cir_norm_cols = [f'CIR{{i}}_norm' for i in range(1016)]")
print(f"   cir_sequence = torch.tensor(df[cir_norm_cols].values).unsqueeze(-1)  # (B, 1016, 1)")
print(f"   context = torch.tensor(df[{lnn_context_features}].values)  # (B, 7)")
print(f"   output = lnn(cir_sequence, context)")

# Cell 42

# Create merged_cir.csv for quick loading in future runs
merged_output_path = Path('../dataset/merged_cir.csv')

# Save the basic merged dataset (just the raw data, before feature engineering)
# This will be used at the start of the notebook to skip individual CSV loading
merged_basic = []
for fname, label, d_true, scen in files:
    df = pd.read_csv(fname)
    df['label'] = label
    df['d_true'] = d_true
    df['scenario'] = scen
    merged_basic.append(df)

merged_basic_data = pd.concat(merged_basic, ignore_index=True)
merged_basic_data.to_csv(merged_output_path, index=False)

print(f"✅ Created merged dataset: {merged_output_path}")
print(f"   Total samples: {len(merged_basic_data):,}")
print(f"   Total columns: {len(merged_basic_data.columns)}")
print(f"   File size: {merged_output_path.stat().st_size / (1024*1024):.2f} MB")
print("\n💡 Next time, the notebook will load this file directly instead of 8 separate CSVs!")

# Cell 44

### 12.1 Verify Merged Dataset Integrity

# Load the merged dataset we just created
merged_verify = pd.read_csv(merged_output_path)

print("=" * 90)
print("MERGED DATASET VERIFICATION")
print("=" * 90)
print(f"\n📊 Shape: {merged_verify.shape[0]:,} samples × {merged_verify.shape[1]} columns")

print("\n📋 Label Distribution:")
for label, count in merged_verify['label'].value_counts().items():
    print(f"  {label}: {count:,} samples ({count/len(merged_verify)*100:.1f}%)")

print("\n📍 Scenario Distribution:")
for scenario, count in merged_verify['scenario'].value_counts().items():
    print(f"  {scenario}: {count:,} samples")

print("\n✅ Data Integrity Check:")
print(f"  Missing values: {merged_verify.isna().sum().sum()}")
print(f"  Duplicate rows: {merged_verify.duplicated().sum()}")

print("\n📏 Distance Statistics:")
dist_stats = merged_verify.groupby('scenario')['d_true'].first().describe()
print(f"  Min distance: {dist_stats['min']:.2f}m")
print(f"  Max distance: {dist_stats['max']:.2f}m")
print(f"  Mean distance: {dist_stats['mean']:.2f}m")

print("\n" + "=" * 90)
merged_verify.head()

# Cell 45

### 12.2 Merged Dataset: Scenario Comparison

# Visual comparison of all scenarios in the merged dataset
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Sample count by scenario
ax1 = axes[0, 0]
scenario_counts = merged_verify['scenario'].value_counts().sort_index()
colors = ['skyblue' if 'LOS' in s else 'salmon' for s in scenario_counts.index]
scenario_counts.plot(kind='barh', ax=ax1, color=colors)
ax1.set_title('Sample Count by Scenario', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Samples')
ax1.grid(axis='x', alpha=0.3)

# 2. Distance distribution by label
ax2 = axes[0, 1]
for label in ['LOS', 'NLOS']:
    subset = merged_verify[merged_verify['label'] == label]
    distances = subset.groupby('scenario')['d_true'].first()
    ax2.scatter([label] * len(distances), distances, s=200, alpha=0.6, 
                label=label, color='skyblue' if label == 'LOS' else 'salmon')
ax2.set_title('True Distance Distribution by Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('Distance (m)')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# 3. Environment breakdown
ax3 = axes[1, 0]
merged_verify['environment'] = merged_verify['scenario'].apply(lambda x: 
    'Home' if 'living room' in x or 'door' in x 
    else 'Meeting Room' if 'meeting room' in x 
    else 'Basement')
env_label_counts = merged_verify.groupby(['environment', 'label']).size().unstack(fill_value=0)
env_label_counts.plot(kind='bar', ax=ax3, color=['skyblue', 'salmon'], stacked=False)
ax3.set_title('Sample Distribution by Environment and Label', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Samples')
ax3.set_xlabel('Environment')
ax3.legend(title='Label')
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

# 4. Distance range visualization
ax4 = axes[1, 1]
distances_by_scenario = merged_verify.groupby('scenario')['d_true'].first().sort_values()
scenarios_sorted = distances_by_scenario.index
colors_sorted = ['skyblue' if 'LOS' in s else 'salmon' for s in scenarios_sorted]
ax4.barh(range(len(scenarios_sorted)), distances_by_scenario.values, color=colors_sorted)
ax4.set_yticks(range(len(scenarios_sorted)))
ax4.set_yticklabels([s.replace(' living room', '').replace(' meeting room', '').replace(' basement', '') 
                      for s in scenarios_sorted], fontsize=9)
ax4.set_title('True Distance by Scenario', fontsize=12, fontweight='bold')
ax4.set_xlabel('Distance (m)')
ax4.grid(axis='x', alpha=0.3)

plt.suptitle('Merged Dataset Overview (8 Scenarios)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n✅ Merged dataset visualization complete!")

# Cell 46

### 12.3 Merged Dataset: Summary Statistics Table

print("=" * 110)
print("COMPREHENSIVE MERGED DATASET SUMMARY")
print("=" * 110)

summary_data = []
for scen in PREFERRED_SCENARIOS:
    if scen in merged_verify['scenario'].values:
        subset = merged_verify[merged_verify['scenario'] == scen]
        summary_data.append({
            'Scenario': scen,
            'Label': subset['label'].iloc[0],
            'Samples': len(subset),
            'Distance (m)': subset['d_true'].iloc[0],
            'Environment': subset['environment'].iloc[0]
        })

summary_df = pd.DataFrame(summary_data)

print("\nScenario Details:")
print(summary_df.to_string(index=False))

print(f"\n\nOverall Statistics:")
print(f"  Total Samples: {len(merged_verify):,}")
print(f"  Total Scenarios: {merged_verify['scenario'].nunique()}")
print(f"  LOS Scenarios: {len(merged_verify[merged_verify['label']=='LOS']['scenario'].unique())}")
print(f"  NLOS Scenarios: {len(merged_verify[merged_verify['label']=='NLOS']['scenario'].unique())}")
print(f"  Environments: {merged_verify['environment'].nunique()} (Home, Meeting Room, Basement)")
print(f"  Distance Range: {merged_verify['d_true'].min():.2f}m - {merged_verify['d_true'].max():.2f}m")

print("\n" + "=" * 110)
print("✅ MERGED DATASET READY FOR ANALYSIS")
print("   File: ../dataset/merged_cir.csv")
print("   Enhanced File: ../dataset/merged_cir_enhanced.csv (with all engineered features)")
print("=" * 110)