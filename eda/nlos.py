import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ================= knobs =================
YMAX = 45
ZOOM_HALF_WIDTH = 12
PRE_WIDTH = 60
PRE_GAP = 8
SHOW_FPIDX = True

INSET_X, INSET_Y = 0.16, -0.06
INSET_W, INSET_H = "38%", "58%"

COL = {
    "nlos": "#d62728",  # NLOS line color
    "f1":   "#ff4fb0",
    "f2":   "#2ca02c",
    "f3":   "#9467bd",
    "tfp":  "#ff7f0e",
    "fpid": "#333333",
}

def sort_cir_cols(cols):
    def key(c):
        m = re.search(r'(\d+)$', c)
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)

def first_crossing(x, thr):
    idx = np.where(x >= thr)[0]
    return int(idx[0]) if len(idx) else None

def local_max_idx(x, left, right):
    left = max(0, left); right = min(len(x), right)
    if right <= left: return left
    return int(left + np.argmax(x[left:right]))

def next_local_peak(x, start, max_span=20):
    i = start + 1
    end = min(len(x) - 2, start + max_span)
    while i <= end:
        if x[i-1] <= x[i] and x[i] >= x[i+1]:
            return i
        i += 1
    return None

def prev_local_peak(x, start, max_span=20):
    i = max(1, start - 1)
    end = max(1, start - max_span)
    while i >= end:
        if x[i-1] <= x[i] and x[i] >= x[i+1]:
            return i
        i -= 1
    return None

def prf_constant_A(prf_mhz):
    if prf_mhz == 16:
        return 115.72
    return 121.74  # default 64 MHz

# 1) load
df = pd.read_csv("uwb_dataset_part1.csv")
cir_cols = sort_cir_cols([c for c in df.columns if c.upper().startswith("CIR")])
if not cir_cols: raise ValueError("No CIR columns found.")
required = ["RXPACC","FP_IDX","CIR_PWR","FP_AMP1","FP_AMP2","FP_AMP3"]
for k in required:
    if k not in df.columns:
        raise ValueError(f"Column {k} missing in CSV")

prf_mhz = int(df.get("PRF_MHZ", pd.Series([64])).iloc[0])
A = prf_constant_A(prf_mhz)

# 2) pick one NLOS row
nlos_df = df[df["NLOS"] == 1.0]
if nlos_df.empty: raise ValueError("No NLOS rows where NLOS == 1.0")
row = nlos_df.iloc[0]

# raw CIR for plotting; normalized only for visuals
cir_raw = row[cir_cols].values.astype(float)
rxpacc = float(row["RXPACC"])
cir_plot = cir_raw / max(rxpacc, 1.0)

n = len(cir_raw); idx = np.arange(n)
fp_idx = int(row["FP_IDX"])
fp_amps = [float(row["FP_AMP1"]), float(row["FP_AMP2"]), float(row["FP_AMP3"])]

print(f"NLOS sample FP_IDX {fp_idx}")
print(f"FP_AMP1 {fp_amps[0]:.2f}  FP_AMP2 {fp_amps[1]:.2f}  FP_AMP3 {fp_amps[2]:.2f}")

# 3) TFP on normalized vector
pre_start = max(0, fp_idx - PRE_WIDTH - PRE_GAP)
pre_end   = max(0, fp_idx - PRE_GAP)
baseline  = cir_plot[pre_start:pre_end] if pre_end > pre_start else cir_plot[:max(fp_idx-1, 1)]
b_med = float(np.median(baseline))
mad   = float(np.median(np.abs(baseline - b_med))) + 1e-12
b_rstd = 1.4826 * mad
fcn_thr = b_med + 8.05 * b_rstd
tfp = first_crossing(cir_plot, fcn_thr) or fp_idx

# 4) features
f1 = int(np.argmax(cir_plot))
left_start = max(0, f1 - 20)
f2 = prev_local_peak(cir_plot, f1, max_span=20)
if f2 is None: f2 = local_max_idx(cir_plot, left_start, f1)
right_end = min(n, f1 + 20)
f3 = next_local_peak(cir_plot, f1, max_span=20)
if f3 is None: f3 = local_max_idx(cir_plot, f1 + 1, right_end)

# 5) Powers from registers (DW1000 equations) + sanity check
EPS = 1e-12
N = rxpacc
C_reg = float(row["CIR_PWR"])
F1, F2, F3 = fp_amps
FP_reg = F1*F1 + F2*F2 + F3*F3

RX_POWER_reg = 10.0*np.log10((C_reg * (2**17)) / (N**2) + EPS) - A
FP_POWER_reg = 10.0*np.log10((FP_reg) / (N**2) + EPS) - A
RFP_reg      = RX_POWER_reg - FP_POWER_reg

ratio_reg = (C_reg * (2**17)) / (FP_reg + EPS)
print(f"Register check: C_reg*2^17 / FP_sum = {ratio_reg:.6g}  (RFP_reg {RFP_reg:.2f} dB)")

use_reg = RFP_reg >= 0.0
if use_reg:
    RX_POWER, FP_POWER, RFP_metric = RX_POWER_reg, FP_POWER_reg, RFP_reg
    rfp_mode = "reg"
else:
    C_cir  = float(np.sum(cir_raw**2))
    w0, w1 = max(0, fp_idx - 1), min(n, fp_idx + 2)
    FP_cir = float(np.sum(cir_raw[w0:w1]**2))
    RFP_metric = 10.0*np.log10((C_cir + EPS) / (FP_cir + EPS))
    RX_POWER = None; FP_POWER = None
    rfp_mode = "cir"
    print(f"Registers inconsistent; using CIR-derived ratio. C_cir/FP_cir = {(C_cir/FP_cir):.6g}, RFP_cir {RFP_metric:.2f} dB")

# rule-of-thumb classification
if RFP_metric < 6:
    rfp_class = "Likely LOS"
elif RFP_metric > 10:
    rfp_class = "Likely NLOS"
else:
    rfp_class = "Ambiguous"

if use_reg:
    print(f"RX_POWER {RX_POWER:.2f} dBm  FP_POWER {FP_POWER:.2f} dBm  RFP {RFP_metric:.2f} dB  {rfp_class}")
else:
    print(f"RFP (CIR-derived, relative) {RFP_metric:.2f} dB  {rfp_class}")

# 6) Plot
fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
ax.plot(idx, cir_plot, color=COL["nlos"], lw=2, label="NLOS CIR")

def mark(axh, x, label, color, dy=10, left=False, size=55):
    i = int(np.clip(round(x), 0, len(cir_plot)-1))
    axh.scatter([x], [cir_plot[i]], s=size, zorder=7, edgecolor="k", color=color)
    axh.annotate(label, (x, cir_plot[i]),
                 textcoords="offset points",
                 xytext=(-14, dy) if left else (8, dy),
                 arrowprops=dict(arrowstyle="->", lw=1, color=color),
                 color=color, fontsize=10, clip_on=True)

mark(ax, tfp,    "TFP",      COL["tfp"],  dy=10,  left=True,  size=70)
mark(ax, fp_idx, "FP_IDX",   COL["fpid"], dy=-14, left=False)
mark(ax, f1,     "F1 / MAX", COL["f1"],   dy=-18, left=False)
mark(ax, f2,     "F2",       COL["f2"],   dy=8,   left=True)
mark(ax, f3,     "F3",       COL["f3"],   dy=8,   left=False)

title_tail = "reg" if use_reg else "cir"
ax.set_title(f"NLOS (FP_IDX {fp_idx}, TFP {tfp}, RFP {RFP_metric:.2f} dB, {rfp_class}, {title_tail})")
ax.set_xlabel("Index")
ax.set_ylabel("Normalized amplitude")
ax.set_xlim(0, n - 1); ax.margins(x=0)
ax.set_ylim(0, YMAX)
ax.grid(alpha=0.25)

# zoom band around FP_IDX
z0 = max(0, fp_idx - ZOOM_HALF_WIDTH)
z1 = min(n, fp_idx + ZOOM_HALF_WIDTH)
ys_local = cir_plot[z0:z1]
pad = 0.07 * (ys_local.max() - ys_local.min() + 1e-9)
band_y0 = max(0.0, ys_local.min() - pad)
band_y1 = min(YMAX, ys_local.max() + pad)
ax.add_patch(Rectangle((z0, band_y0), z1 - z0, band_y1 - band_y0,
                       lw=1.5, ec="black", fc="wheat", alpha=0.18))
if SHOW_FPIDX:
    ax.vlines(fp_idx, band_y0, band_y1, linestyles="--",
              colors=COL["fpid"], lw=1.2, alpha=0.9, label="FP_IDX")

legend_handles = [
    Line2D([0], [0], color=COL["nlos"], lw=2, label="NLOS CIR"),
    Line2D([0], [0], color=COL["fpid"], lw=1.2, ls="--", label="FP_IDX"),
    Line2D([0], [0], marker='o', color='w', label='F2', markerfacecolor=COL["f2"], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='F3', markerfacecolor=COL["f3"], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='F1 / MAX', markerfacecolor=COL["f1"], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='TFP', markerfacecolor=COL["tfp"], markeredgecolor='k', markersize=8),
]
ax.legend(handles=legend_handles, loc="upper right", ncols=2, fontsize=9)

# inset centered on F1, y from 0
axins = inset_axes(ax, width=INSET_W, height=INSET_H, loc="upper left",
                   bbox_to_anchor=(INSET_X, INSET_Y, 1.0, 1.0),
                   bbox_transform=ax.transAxes, borderpad=0.0)
axins.plot(idx, cir_plot, color=COL["nlos"], lw=2)
if SHOW_FPIDX:
    axins.axvline(fp_idx, ls="--", lw=1, color=COL["fpid"])
z0_i = max(0, f1 - ZOOM_HALF_WIDTH); z1_i = min(n, f1 + ZOOM_HALF_WIDTH)
ys_local_i = cir_plot[z0_i:z1_i]
pad_i = 0.07 * (ys_local_i.max() - ys_local_i.min() + 1e-9)
band_y0_i = 0.0
band_y1_i = min(YMAX, ys_local_i.max() + pad_i)
mark(axins, tfp,    "TFP",      COL["tfp"],  dy=10,  left=True,  size=70)
mark(axins, fp_idx, "FP_IDX",   COL["fpid"], dy=-12, left=False)
mark(axins, f1,     "Max & F1", COL["f1"],   dy=-16, left=False)
mark(axins, f2,     "F2",       COL["f2"],   dy=7,   left=True)
mark(axins, f3,     "F3",       COL["f3"],   dy=7,   left=False)
axins.set_xlim(z0_i, z1_i)
axins.set_ylim(band_y0_i, band_y1_i)
axins.set_xlabel("Index", fontsize=8)
axins.set_ylabel("Amp", fontsize=8)
axins.tick_params(axis="both", labelsize=8)
axins.grid(alpha=0.25)

start_in_inset = (0.98, 0.5)
target_x = z0_i
target_y = 0.5 * (band_y0_i + band_y1_i)
conn = ConnectionPatch(xyA=start_in_inset, coordsA=axins.transAxes,
                       xyB=(target_x, target_y), coordsB=ax.transData,
                       arrowstyle="->", lw=1.4, color="#444")
ax.add_artist(conn)

plt.savefig("nlos_cir_register_checked.png", dpi=300, bbox_inches="tight")
plt.show()
