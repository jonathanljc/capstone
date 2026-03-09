"""
Update all 4 Stage 3 notebooks to use pre-labeled is_correctable from CSV
instead of recomputing on the fly.

Changes:
1. Simplify load_correctable_nlos() — read is_correctable column
2. Simplify unseen eval — read is_correctable from unseen CSV for ground truth
3. Keep get_roi_alignment (needed for CIR preprocessing)
4. Remove count_peaks and compute_bounce_dominance (no longer needed)
"""

import json

# ── New data loading cell (common to all) ──────────────────────────────────
NEW_DATA_LOADING = '''# ==========================================
# ROI ALIGNMENT (reused from Stage 1/2)
# ==========================================
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


# ==========================================
# LOAD NLOS DATA + FILTER TO CORRECTABLE (using pre-labeled CSV)
# ==========================================
def load_correctable_nlos(filepath="../dataset/channels/combined_uwb_dataset.csv"):
    """
    Load NLOS samples filtered to correctable using pre-labeled columns
    (is_correctable, num_peaks, bounce_dominance) from the dataset CSV.
    """
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    cir_cols = sorted(
        [c for c in df.columns if c.startswith('CIR')],
        key=lambda x: int(x.replace('CIR', ''))
    )

    # Filter to correctable NLOS using pre-labeled column
    corr_df = df[(df["Label"] == 1) & (df["is_correctable"] == 1)].reset_index(drop=True)
    total_nlos = (df["Label"] == 1).sum()
    print(f"  Total samples: {len(df)}, NLOS: {total_nlos}, Correctable: {len(corr_df)} ({100*len(corr_df)/total_nlos:.1f}%)")

    raw_sigs = []
    leading_edges = []
    d_hardware_list = []
    d_direct_list = []
    d_bounce_list = []
    source_files = []

    for idx, row in corr_df.iterrows():
        sig = pd.to_numeric(row[cir_cols], errors='coerce').fillna(0).astype(float).values

        # RXPACC normalization
        rxpacc_col = 'RXPACC' if 'RXPACC' in row.index else 'RX_PACC'
        rxpacc = float(row.get(rxpacc_col, 128.0))
        if rxpacc > 0:
            sig = sig / rxpacc

        le = get_roi_alignment(sig)
        raw_sigs.append(sig)
        leading_edges.append(le)

        d_hardware_list.append(float(row['Distance']))
        d_direct_list.append(float(row['d_direct']))
        d_bounce_list.append(float(row['d_bounce']))
        source_files.append(str(row.get('Source_File', '')))

    leading_edges = np.array(leading_edges)
    d_hardware = np.array(d_hardware_list, dtype=float)
    d_direct = np.array(d_direct_list, dtype=float)
    d_bounce = np.array(d_bounce_list, dtype=float)

    # Ranging error target
    ranging_error = d_hardware - d_direct

    # Scenario groups
    groups = []
    for sf in source_files:
        match = re.search(r'([\\d.]+)m_nlos', sf)
        groups.append(match.group(1) + 'm' if match else 'unknown')
    groups = np.array(groups)

    print(f"\\n  Ranging error target (Distance - d_direct):")
    print(f"    Mean: {ranging_error.mean():.3f}m, Std: {ranging_error.std():.3f}m")
    print(f"    Min: {ranging_error.min():.3f}m, Max: {ranging_error.max():.3f}m")
    print(f"\\n  Per scenario:")
    for g in sorted(set(groups)):
        mask = groups == g
        re_g = ranging_error[mask]
        print(f"    {g}: n={int(mask.sum())}, RE mean={re_g.mean():.3f}m, std={re_g.std():.3f}m")

    return (np.array(raw_sigs), leading_edges,
            d_hardware, d_direct, d_bounce, ranging_error, groups)


raw_sigs, leading_edges, d_hardware, d_direct, d_bounce, ranging_error, groups = load_correctable_nlos()
'''

# ── New unseen eval template (encoder-specific parts differ) ───────────────
def make_unseen_eval(encoder_name, encoder_var, forward_call, preprocess_func, tensor_reshape):
    """Generate unseen eval cell code for a specific encoder."""
    return f'''# ==========================================
# CHAINED PIPELINE UNSEEN EVALUATION
# Stage 1 (NLOS?) → Stage 2 (Correctable?) → Stage 3 (Error correction)
# ==========================================
unseen_filepath = "../dataset/channels/unseen_dataset.csv"

# ── 1. Load ALL unseen data (no filtering) ──
print(f"Loading ALL unseen data: {{unseen_filepath}}")
df_unseen = pd.read_csv(unseen_filepath)
cir_cols = sorted([c for c in df_unseen.columns if c.startswith('CIR')],
                  key=lambda x: int(x.replace('CIR', '')))

all_sigs, all_les = [], []
all_d_hw, all_d_direct, all_labels, all_source = [], [], [], []

for idx, row in df_unseen.iterrows():
    sig = pd.to_numeric(row[cir_cols], errors='coerce').fillna(0).astype(float).values
    rxpacc_col = 'RXPACC' if 'RXPACC' in row.index else 'RX_PACC'
    rxpacc = float(row.get(rxpacc_col, 128.0))
    if rxpacc > 0:
        sig = sig / rxpacc
    le = get_roi_alignment(sig)
    all_sigs.append(sig)
    all_les.append(le)
    all_d_hw.append(float(row['Distance']))
    all_d_direct.append(float(row['d_direct']))
    all_labels.append(int(row['Label']))
    all_source.append(str(row.get('Source_File', '')))

all_d_hw = np.array(all_d_hw)
all_d_direct = np.array(all_d_direct)
all_labels = np.array(all_labels)

# Ground truth from pre-labeled CSV columns
gt_nlos = all_labels == 1
gt_correctable = (df_unseen["Label"] == 1) & (df_unseen["is_correctable"] == 1)
gt_correctable = gt_correctable.values

print(f"  Total unseen: {{len(df_unseen)}}")
print(f"  Ground truth NLOS: {{gt_nlos.sum()}}")
print(f"  Ground truth correctable: {{gt_correctable.sum()}}")

# ── 2. Preprocess all CIR for encoder ──
cir_sequences = []
for i in range(len(all_sigs)):
    crop = {preprocess_func}(all_sigs[i], all_les[i])
    cir_sequences.append(crop)

cir_tensor = torch.tensor(
    np.array(cir_sequences){tensor_reshape},
    dtype=torch.float32
).to(device)

# ── 3. Stage 1: Predict NLOS ──
print(f"\\n--- Stage 1: LOS/NLOS Classification ---")
stage1_probs = []
with torch.no_grad():
    for i in range(0, len(cir_tensor), 256):
        batch = cir_tensor[i:i+256]
        {forward_call}
        stage1_probs.append(pred.cpu().numpy())
stage1_probs = np.concatenate(stage1_probs).flatten()
predicted_nlos = stage1_probs >= 0.5
predicted_nlos_idx = np.where(predicted_nlos)[0]

print(f"  Predicted NLOS: {{predicted_nlos.sum()}} / {{len(df_unseen)}}")
print(f"  Actual NLOS in predicted set: {{all_labels[predicted_nlos_idx].sum()}} / {{predicted_nlos.sum()}}")
s1_nlos_acc = (predicted_nlos == gt_nlos).mean()
print(f"  Stage 1 accuracy on unseen: {{s1_nlos_acc:.4f}}")

# ── 4. Stage 2: Predict Correctable among predicted NLOS ──
print(f"\\n--- Stage 2: Correctable Classification ---")
stage2_rf = joblib.load("stage2_bounce_rf.joblib")

nlos_embeddings = []
with torch.no_grad():
    for i in range(0, len(predicted_nlos_idx), 256):
        batch_idx = predicted_nlos_idx[i:i+256]
        batch = cir_tensor[batch_idx]
        emb = {encoder_var}.embed(batch)
        nlos_embeddings.append(emb.cpu().numpy())
nlos_embeddings = np.vstack(nlos_embeddings)

stage2_preds = stage2_rf.predict(nlos_embeddings)
predicted_correctable_mask = stage2_preds == 1
correctable_idx = predicted_nlos_idx[predicted_correctable_mask]

print(f"  Predicted correctable: {{predicted_correctable_mask.sum()}} / {{len(predicted_nlos_idx)}} predicted NLOS")
print(f"  Actually NLOS in correctable set: {{all_labels[correctable_idx].sum()}} / {{len(correctable_idx)}}")
print(f"  Actually correctable (GT) in set: {{gt_correctable[correctable_idx].sum()}} / {{len(correctable_idx)}}")

# ── 5. Stage 3: Predict ranging error for pipeline-selected samples ──
print(f"\\n--- Stage 3: Ranging Error Prediction ---")
corr_embeddings = []
with torch.no_grad():
    for i in range(0, len(correctable_idx), 256):
        batch_idx = correctable_idx[i:i+256]
        batch = cir_tensor[batch_idx]
        emb = {encoder_var}.embed(batch)
        corr_embeddings.append(emb.cpu().numpy())
corr_embeddings = np.vstack(corr_embeddings)

predicted_errors = rf_model.predict(corr_embeddings)

# ── 6. Evaluate: pipeline-selected samples ──
d_hw_pipe = all_d_hw[correctable_idx]
d_dir_pipe = all_d_direct[correctable_idx]
labels_pipe = all_labels[correctable_idx]
gt_corr_pipe = gt_correctable[correctable_idx]

raw_errors_pipe = d_hw_pipe - d_dir_pipe
corrected_distances_pipe = d_hw_pipe - predicted_errors
corrected_errors_pipe = corrected_distances_pipe - d_dir_pipe

# Scenario groups for pipeline samples
groups_pipe = []
for idx in correctable_idx:
    sf = all_source[idx]
    match = re.search(r'([\\d.]+)m_(nlos|los)', sf)
    groups_pipe.append(match.group(1) + 'm_' + match.group(2) if match else 'unknown')
groups_pipe = np.array(groups_pipe)

print(f"\\n{{'='*60}}")
print(f"CHAINED PIPELINE RESULTS (unseen data)")
print(f"{{'='*60}}")
print(f"  Total unseen samples:        {{len(df_unseen)}}")
print(f"  Stage 1 → predicted NLOS:    {{predicted_nlos.sum()}}")
print(f"  Stage 2 → predicted correct: {{len(correctable_idx)}}")
print(f"  Pipeline throughput:          {{len(correctable_idx)}}/{{len(df_unseen)}} ({{100*len(correctable_idx)/len(df_unseen):.1f}}%)")
print()

# Only evaluate correction on samples that are actually NLOS
actually_nlos_mask = labels_pipe == 1
print(f"  Of {{len(correctable_idx)}} pipeline-selected samples:")
print(f"    Actually NLOS:        {{actually_nlos_mask.sum()}}")
print(f"    Actually LOS (FP):    {{(~actually_nlos_mask).sum()}}")
print(f"    Actually correctable: {{gt_corr_pipe.sum()}}")

if actually_nlos_mask.sum() > 0:
    raw_mae_nlos = np.abs(raw_errors_pipe[actually_nlos_mask]).mean()
    corr_mae_nlos = np.abs(corrected_errors_pipe[actually_nlos_mask]).mean()
    print(f"\\n  NLOS samples in pipeline (n={{actually_nlos_mask.sum()}}):")
    print(f"    Before ML:  MAE = {{raw_mae_nlos:.4f}}m")
    print(f"    After ML:   MAE = {{corr_mae_nlos:.4f}}m")
    if corr_mae_nlos > 0:
        print(f"    Improvement: {{raw_mae_nlos/corr_mae_nlos:.1f}}x")

    # Per-scenario breakdown (NLOS only)
    print(f"\\n  Per-scenario (NLOS in pipeline):")
    for g in sorted(set(groups_pipe[actually_nlos_mask])):
        mask = (groups_pipe == g) & actually_nlos_mask
        if mask.sum() > 0:
            raw_g = np.abs(raw_errors_pipe[mask]).mean()
            corr_g = np.abs(corrected_errors_pipe[mask]).mean()
            print(f"    {{g}}: n={{mask.sum()}}, before={{raw_g:.4f}}m, after={{corr_g:.4f}}m")

# False positive impact: LOS samples incorrectly "corrected"
if (~actually_nlos_mask).sum() > 0:
    los_raw = np.abs(raw_errors_pipe[~actually_nlos_mask]).mean()
    los_corrected = np.abs(corrected_errors_pipe[~actually_nlos_mask]).mean()
    print(f"\\n  LOS false positives (n={{(~actually_nlos_mask).sum()}}):")
    print(f"    Before ML (already small): MAE = {{los_raw:.4f}}m")
    print(f"    After ML (wrongly corrected): MAE = {{los_corrected:.4f}}m")

# Overall: all pipeline-selected samples
overall_raw_mae = np.abs(raw_errors_pipe).mean()
overall_corr_mae = np.abs(corrected_errors_pipe).mean()
print(f"\\n  ALL pipeline-selected (n={{len(correctable_idx)}}):")
print(f"    Before ML:  MAE = {{overall_raw_mae:.4f}}m")
print(f"    After ML:   MAE = {{overall_corr_mae:.4f}}m")
if overall_corr_mae > 0:
    print(f"    Improvement: {{overall_raw_mae/overall_corr_mae:.1f}}x")'''


# ── Notebook configs ───────────────────────────────────────────────────────
NOTEBOOKS = {
    "capstone/lnn/stage3_ranging_error.ipynb": {
        "data_cell_idx": 3,
        "unseen_cell_idx": 14,
        "encoder_name": "LNN",
        "encoder_var": "lnn_encoder",
        "forward_call": "pred, _, _ = lnn_encoder(batch)  # LNN returns (pred, tau_los, tau_nlos)",
        "preprocess_func": "preprocess_cir_for_lnn",
        "tensor_reshape": ".reshape(-1, STAGE1_CONFIG['total_len'], 1)",
    },
    "capstone/lstm/stage3_ranging_error.ipynb": {
        "data_cell_idx": 3,
        "unseen_cell_idx": 14,
        "encoder_name": "LSTM",
        "encoder_var": "lstm_encoder",
        "forward_call": "pred = lstm_encoder(batch)  # LSTM returns pred directly",
        "preprocess_func": "preprocess_cir_for_lstm",
        "tensor_reshape": ".reshape(-1, STAGE1_CONFIG['total_len'], 1)",
    },
    "capstone/cnn/stage3_ranging_error.ipynb": {
        "data_cell_idx": 3,
        "unseen_cell_idx": 14,
        "encoder_name": "CNN",
        "encoder_var": "cnn_encoder",
        "forward_call": "pred = cnn_encoder(batch)  # CNN returns pred directly",
        "preprocess_func": "preprocess_cir_for_cnn",
        "tensor_reshape": ".reshape(-1, 1, STAGE1_CONFIG['total_len'])",
    },
    "capstone/bert/stage3_ranging_error.ipynb": {
        "data_cell_idx": 3,
        "unseen_cell_idx": 14,
        "encoder_name": "BERT",
        "encoder_var": "bert_encoder",
        "forward_call": "pred = bert_encoder(batch)  # BERT returns pred directly",
        "preprocess_func": "preprocess_cir_for_bert",
        "tensor_reshape": ".reshape(-1, STAGE1_CONFIG['total_len'], 1)",
    },
}


# ── Apply changes ──────────────────────────────────────────────────────────
for nb_path, cfg in NOTEBOOKS.items():
    print(f"\nUpdating: {nb_path}")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update data loading cell
    cell = nb['cells'][cfg['data_cell_idx']]
    old_src = ''.join(cell['source'])
    cell['source'] = [NEW_DATA_LOADING]
    cell['outputs'] = []
    print(f"  Cell {cfg['data_cell_idx']} (data loading): updated")

    # Update unseen eval cell
    unseen_code = make_unseen_eval(
        cfg['encoder_name'], cfg['encoder_var'],
        cfg['forward_call'], cfg['preprocess_func'],
        cfg['tensor_reshape']
    )
    cell = nb['cells'][cfg['unseen_cell_idx']]
    cell['source'] = [unseen_code]
    cell['outputs'] = []
    print(f"  Cell {cfg['unseen_cell_idx']} (unseen eval): updated")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {nb_path}")


print("\n✓ All 4 Stage 3 notebooks updated!")
print("  - load_correctable_nlos() now reads is_correctable from CSV")
print("  - Unseen eval reads gt_correctable from CSV (no recomputation)")
print("  - Removed count_peaks() and compute_bounce_dominance() from data loading")
