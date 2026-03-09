"""Add unseen evaluation section to all 4 Stage 3 notebooks."""
import json

NOTEBOOKS = {
    "capstone/lnn/stage3_ranging_error.ipynb": {
        "name": "LNN",
        "encoder_var": "lnn_encoder",
        "preprocess_func": "preprocess_cir_for_lnn",
        "tensor_reshape": '.reshape(-1, STAGE1_CONFIG["total_len"], 1)',
    },
    "capstone/lstm/stage3_ranging_error.ipynb": {
        "name": "LSTM",
        "encoder_var": "lstm_encoder",
        "preprocess_func": "preprocess_cir_for_lstm",
        "tensor_reshape": '.reshape(-1, STAGE1_CONFIG["total_len"], 1)',
    },
    "capstone/cnn/stage3_ranging_error.ipynb": {
        "name": "CNN",
        "encoder_var": "cnn_encoder",
        "preprocess_func": "preprocess_cir_for_cnn",
        "tensor_reshape": '.reshape(-1, 1, STAGE1_CONFIG["total_len"])',
    },
    "capstone/bert/stage3_ranging_error.ipynb": {
        "name": "BERT",
        "encoder_var": "bert_encoder",
        "preprocess_func": "preprocess_cir_for_bert",
        "tensor_reshape": '.reshape(-1, STAGE1_CONFIG["total_len"], 1)',
    },
}


def make_unseen_code(cfg):
    return f'''# ==========================================
# UNSEEN EVALUATION (Oracle-filtered correctable NLOS)
# Fair comparison: same correctable samples across all models
# ==========================================
unseen_filepath = "../dataset/channels/unseen_dataset.csv"

print(f"Loading unseen data: {{unseen_filepath}}")
df_unseen = pd.read_csv(unseen_filepath)
cir_cols_u = sorted([c for c in df_unseen.columns if c.startswith('CIR')],
                    key=lambda x: int(x.replace('CIR', '')))

# Filter to correctable NLOS using pre-labeled column
unseen_corr = df_unseen[(df_unseen["Label"] == 1) & (df_unseen["is_correctable"] == 1)].reset_index(drop=True)
total_nlos_u = (df_unseen["Label"] == 1).sum()
print(f"  Total unseen: {{len(df_unseen)}}, NLOS: {{total_nlos_u}}, Correctable: {{len(unseen_corr)}}")

# Extract CIR, preprocess, and get distances
unseen_sigs, unseen_les = [], []
unseen_d_hw, unseen_d_direct = [], []
unseen_source = []

for idx, row in unseen_corr.iterrows():
    sig = pd.to_numeric(row[cir_cols_u], errors='coerce').fillna(0).astype(float).values
    rxpacc = float(row.get('RXPACC', 128.0))
    if rxpacc > 0:
        sig = sig / rxpacc
    le = get_roi_alignment(sig)
    unseen_sigs.append(sig)
    unseen_les.append(le)
    unseen_d_hw.append(float(row['Distance']))
    unseen_d_direct.append(float(row['d_direct']))
    unseen_source.append(str(row.get('Source_File', '')))

unseen_d_hw = np.array(unseen_d_hw)
unseen_d_direct = np.array(unseen_d_direct)
unseen_ranging_error = unseen_d_hw - unseen_d_direct

# Scenario groups
unseen_groups = []
for sf in unseen_source:
    match = re.search(r'([\\d.]+)m_nlos', sf)
    unseen_groups.append(match.group(1) + 'm' if match else 'unknown')
unseen_groups = np.array(unseen_groups)

# Preprocess CIR and extract embeddings
encoder = {cfg["encoder_var"]}
unseen_cir_seqs = []
for i in range(len(unseen_sigs)):
    crop = {cfg["preprocess_func"]}(unseen_sigs[i], unseen_les[i])
    unseen_cir_seqs.append(crop)

unseen_tensor = torch.tensor(
    np.array(unseen_cir_seqs){cfg["tensor_reshape"]},
    dtype=torch.float32
).to(device)

unseen_embeddings = []
with torch.no_grad():
    for i in range(0, len(unseen_tensor), 256):
        batch = unseen_tensor[i:i+256]
        emb = encoder.embed(batch)
        unseen_embeddings.append(emb.cpu().numpy())
unseen_embeddings = np.vstack(unseen_embeddings)

# Predict ranging error
unseen_pred_errors = rf_model.predict(unseen_embeddings)
unseen_corrected = unseen_d_hw - unseen_pred_errors
unseen_corr_errors = unseen_corrected - unseen_d_direct

# Metrics
unseen_mae = mean_absolute_error(unseen_ranging_error, unseen_pred_errors)
unseen_rmse = np.sqrt(mean_squared_error(unseen_ranging_error, unseen_pred_errors))
unseen_r2 = r2_score(unseen_ranging_error, unseen_pred_errors)

raw_mae_unseen = np.abs(unseen_ranging_error).mean()
corr_mae_unseen = np.abs(unseen_corr_errors).mean()

print(f"\\n{{'='*60}}")
print(f"UNSEEN EVALUATION -- {cfg["name"]} Stage 3 (Oracle Correctable)")
print(f"{{'='*60}}")
print(f"  Correctable NLOS samples: {{len(unseen_corr)}}")
print(f"  Regressor MAE:  {{unseen_mae:.4f}}m")
print(f"  Regressor RMSE: {{unseen_rmse:.4f}}m")
print(f"  Regressor R2:   {{unseen_r2:.4f}}")
print(f"\\n  Distance correction:")
print(f"    Before ML: MAE = {{raw_mae_unseen:.4f}}m")
print(f"    After ML:  MAE = {{corr_mae_unseen:.4f}}m")
if corr_mae_unseen > 0:
    print(f"    Improvement: {{raw_mae_unseen/corr_mae_unseen:.1f}}x")

print(f"\\n  Per scenario:")
for g in sorted(set(unseen_groups)):
    mask = unseen_groups == g
    if mask.sum() > 0:
        raw_g = np.abs(unseen_ranging_error[mask]).mean()
        corr_g = np.abs(unseen_corr_errors[mask]).mean()
        mae_g = np.abs(unseen_ranging_error[mask] - unseen_pred_errors[mask]).mean()
        print(f"    {{g}}: n={{mask.sum()}}, before={{raw_g:.4f}}m, after={{corr_g:.4f}}m, reg_MAE={{mae_g:.4f}}m")

# ── Visualization ──
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Predicted vs Actual
ax = axs[0]
for g in sorted(set(unseen_groups)):
    mask = unseen_groups == g
    ax.scatter(unseen_ranging_error[mask], unseen_pred_errors[mask], s=25, alpha=0.6, label=g)
lims = [min(unseen_ranging_error.min(), unseen_pred_errors.min()) - 0.2,
        max(unseen_ranging_error.max(), unseen_pred_errors.max()) + 0.2]
ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.5, label='Perfect')
ax.set_xlabel('Actual Ranging Error (m)')
ax.set_ylabel('Predicted Ranging Error (m)')
ax.set_title(f'Pred vs Actual (R2={{unseen_r2:.4f}})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Residual histogram
ax = axs[1]
residuals = unseen_pred_errors - unseen_ranging_error
ax.hist(residuals, bins=40, color='#e74c3c', edgecolor='white', alpha=0.8)
ax.axvline(0, color='black', ls='--', lw=1.5)
ax.axvline(residuals.mean(), color='blue', ls='--', lw=1.5,
           label=f'Mean bias: {{residuals.mean():.3f}}m')
ax.set_xlabel('Residual (predicted - actual, m)')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Per-scenario before vs after
ax = axs[2]
before_maes, after_maes, labels_s = [], [], []
for g in sorted(set(unseen_groups)):
    mask = unseen_groups == g
    if mask.sum() > 0:
        before_maes.append(np.abs(unseen_ranging_error[mask]).mean())
        after_maes.append(np.abs(unseen_corr_errors[mask]).mean())
        labels_s.append(g)
x_pos = np.arange(len(labels_s))
w = 0.35
ax.bar(x_pos - w/2, before_maes, w, color='#e74c3c', alpha=0.7, label='Before ML')
ax.bar(x_pos + w/2, after_maes, w, color='#2ecc71', alpha=0.7, label='After ML')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_s, rotation=15)
for i in range(len(x_pos)):
    if before_maes[i] > 0 and after_maes[i] > 0:
        imp = before_maes[i] / after_maes[i]
        ax.text(x_pos[i], max(before_maes[i], after_maes[i]) + 0.1,
                f'{{imp:.1f}}x', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('MAE (m)')
ax.set_title('Before vs After Correction')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Stage 3 Unseen Evaluation -- {cfg["name"]} (Oracle Correctable, n={{len(unseen_corr)}})',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()'''


for nb_path, cfg in NOTEBOOKS.items():
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    md_cell = {
        "cell_type": "markdown",
        "id": "md-unseen",
        "metadata": {},
        "source": [
            "---\n",
            "## Section 7: Unseen Dataset Evaluation\n",
            "\n",
            "Oracle-filtered correctable NLOS from unseen dataset (same samples for all models = fair comparison)."
        ],
    }
    code_cell = {
        "cell_type": "code",
        "id": "code-unseen",
        "metadata": {},
        "source": [make_unseen_code(cfg)],
        "outputs": [],
        "execution_count": None,
    }

    # Insert after cell 12 (diagnostics)
    nb['cells'].insert(13, md_cell)
    nb['cells'].insert(14, code_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Added unseen eval to {nb_path} (now {len(nb['cells'])} cells)")

print("\nDone!")
