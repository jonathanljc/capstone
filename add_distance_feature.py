"""Add Distance (d_hardware) as extra RF feature to all 4 Stage 3 notebooks."""
import json
import re

NOTEBOOKS = [
    "capstone/lnn/stage3_ranging_error.ipynb",
    "capstone/lstm/stage3_ranging_error.ipynb",
    "capstone/cnn/stage3_ranging_error.ipynb",
    "capstone/bert/stage3_ranging_error.ipynb",
]

def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])

        # --- Patch 1: RF training cell (add Distance to X_all) ---
        if 'X_all = embeddings' in src and 'rf_model.fit' in src:
            new_src = src

            # Add Distance feature to X_all
            new_src = new_src.replace(
                'X_all = embeddings',
                'X_all = np.hstack([embeddings, d_hardware.reshape(-1, 1)])  # embeddings + Distance'
            )

            # Update print to mention +1 Distance
            # Find the FEATURE_DIM print line and update it
            new_src = re.sub(
                r'print\(f"Input features: \{FEATURE_DIM\} dimensions \([^)]+\)"\)',
                'print(f"Input features: {FEATURE_DIM} dimensions (encoder embeddings + Distance)")',
                new_src
            )

            cell['source'] = [new_src]
            patched += 1
            print(f"  Patched RF training cell")

        # --- Patch 2: Unseen eval cell (add Distance to unseen features) ---
        if 'unseen_pred_errors = rf_model.predict(unseen_embeddings)' in src:
            new_src = src.replace(
                'unseen_pred_errors = rf_model.predict(unseen_embeddings)',
                '# Add Distance feature (same as training)\n'
                'unseen_features = np.hstack([unseen_embeddings, unseen_d_hw.reshape(-1, 1)])\n'
                'unseen_pred_errors = rf_model.predict(unseen_features)'
            )
            cell['source'] = [new_src]
            patched += 1
            print(f"  Patched unseen eval cell")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  Total patches: {patched}")
    return patched


for nb_path in NOTEBOOKS:
    print(f"\nProcessing: {nb_path}")
    patch_notebook(nb_path)

print("\nDone! All 4 Stage 3 notebooks updated with Distance feature.")
