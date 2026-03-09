"""
Add num_peaks, bounce_dominance, and is_correctable columns to dataset CSVs.
Uses the same signal processing logic as Stage 2 (LNN reference implementation).
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ── CONFIG (matches Stage 2) ──────────────────────────────────────────────
CONFIG = {
    "search_start": 740,
    "search_end": 810,
    "peak_prominence": 0.20,
    "peak_min_distance": 5,
    "dominant_path_max_peaks": 2,
    "bounce_search_window": 3,
    "dominance_threshold": 0.50,
}

# ── Signal processing helpers (from LNN Stage 2) ─────────────────────────

def get_roi_alignment(sig, search_start=CONFIG["search_start"],
                      search_end=CONFIG["search_end"]):
    """Find leading edge by backtracking from peak."""
    region = sig[search_start:search_end]
    if len(region) == 0:
        return np.argmax(sig)

    peak_local = np.argmax(region)
    peak_idx = search_start + peak_local
    peak_val = sig[peak_idx]

    noise_section = sig[:search_start]
    if len(noise_section) > 10:
        noise_mean = np.mean(noise_section)
        noise_std = np.std(noise_section)
        threshold = max(noise_mean + 3 * noise_std, 0.05 * peak_val)
    else:
        threshold = 0.05 * peak_val

    leading_edge = peak_idx
    for i in range(peak_idx, max(search_start - 20, 0), -1):
        if sig[i] < threshold:
            leading_edge = i + 1
            break

    return leading_edge


def count_peaks_in_roi(sig, leading_edge, config=CONFIG):
    """Count prominent peaks in CIR ROI (120 samples from leading edge)."""
    roi_start = max(0, leading_edge - 5)
    roi_end = min(len(sig), leading_edge + 120)
    roi = sig[roi_start:roi_end]
    if len(roi) == 0 or np.max(roi) == 0:
        return 0
    roi_norm = roi / np.max(roi)
    peaks, _ = find_peaks(
        roi_norm,
        prominence=config["peak_prominence"],
        distance=config["peak_min_distance"]
    )
    return len(peaks)


def compute_bounce_dominance(sig, leading_edge, bounce_path_idx,
                              window=CONFIG["bounce_search_window"]):
    """
    Amplitude ratio: peak near bounce position / strongest peak in ROI.
    Returns 0-1. High = bounce peak is dominant.
    """
    roi_start = max(0, leading_edge - 5)
    roi_end = min(len(sig), leading_edge + 120)
    roi = sig[roi_start:roi_end]

    if len(roi) == 0 or np.max(roi) == 0:
        return 0.0

    strongest_amp = float(np.max(roi))

    bounce_idx = int(round(bounce_path_idx))
    b_start = max(0, bounce_idx - window)
    b_end = min(len(sig), bounce_idx + window + 1)

    if b_start >= b_end:
        return 0.0

    bounce_amp = float(np.max(sig[b_start:b_end]))
    return bounce_amp / strongest_amp


# ── Process a CSV ─────────────────────────────────────────────────────────

def add_labels_to_csv(filepath):
    print(f"\nProcessing: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Total rows: {len(df)}")

    # Detect CIR columns
    cir_cols = sorted(
        [c for c in df.columns if c.startswith('CIR')],
        key=lambda x: int(x.replace('CIR', ''))
    )
    print(f"  CIR columns: {len(cir_cols)} ({cir_cols[0]} to {cir_cols[-1]})")

    num_peaks_list = []
    bounce_dom_list = []
    is_correctable_list = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"  Processing row {idx}/{len(df)}...")

        label = row["Label"]
        sig = row[cir_cols].values.astype(float)

        # Normalize by RXPACC
        rxpacc = float(row.get("RXPACC", 1.0))
        if rxpacc > 0:
            sig = sig / rxpacc

        leading_edge = get_roi_alignment(sig)
        n_peaks = count_peaks_in_roi(sig, leading_edge)
        num_peaks_list.append(n_peaks)

        # Bounce dominance only for NLOS (Label==1)
        if label == 1:
            bpi = float(row.get("bounce_path_idx", np.nan))
            if np.isnan(bpi):
                bounce_dom_list.append(0.0)
            else:
                bd = compute_bounce_dominance(sig, leading_edge, bpi)
                bounce_dom_list.append(bd)
        else:
            # LOS: no bounce path
            bounce_dom_list.append(0.0)

        # Correctable: NLOS + bounce dominant + few peaks
        if label == 1:
            is_corr = int(
                bounce_dom_list[-1] >= CONFIG["dominance_threshold"]
                and n_peaks <= CONFIG["dominant_path_max_peaks"]
            )
        else:
            is_corr = 0  # LOS samples are not "correctable NLOS"

        is_correctable_list.append(is_corr)

    df["num_peaks"] = num_peaks_list
    df["bounce_dominance"] = np.round(bounce_dom_list, 6)
    df["is_correctable"] = is_correctable_list

    # Summary
    nlos = df[df["Label"] == 1]
    corr = nlos[nlos["is_correctable"] == 1]
    print(f"\n  NLOS samples: {len(nlos)}")
    print(f"  Correctable:  {len(corr)} ({100*len(corr)/len(nlos):.1f}%)")
    print(f"  Challenging:  {len(nlos)-len(corr)} ({100*(len(nlos)-len(corr))/len(nlos):.1f}%)")

    df.to_csv(filepath, index=False)
    print(f"  Saved to: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    add_labels_to_csv("channels/combined_uwb_dataset.csv")
    add_labels_to_csv("channels/unseen_dataset.csv")
    print("\nDone! Columns added: num_peaks, bounce_dominance, is_correctable")
