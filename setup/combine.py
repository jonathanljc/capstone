import pandas as pd
import glob
import os

def combine_all_datasets(search_path=CHANNEL_DIR):
    """
    Finds all LOS/NLOS csv files, labels them, and combines them into one.
    """
    all_files = sorted(glob.glob(os.path.join(search_path, "*.csv")))
    
    if not all_files:
        print(f"No CSV files found in {search_path}!")
        return None

    print(f"Found {len(all_files)} files in {search_path}. Combining...")

    combined_data = []

    for filename in all_files:
        try:
            # 2. Determine Label
            fname = os.path.basename(filename).lower()
            if 'nlos' in fname:
                label = 1  # NLOS
                label_str = "NLOS"
            elif 'los' in fname:
                label = 0  # LOS
                label_str = "LOS"
            else:
                print(f"  Skipping {fname}: Cannot determine LOS/NLOS from name.")
                continue

            # 3. Load Data
            df = pd.read_csv(filename)
            
            # Add metadata columns for tracking
            df['Label'] = label
            df['Source_File'] = fname
            
            # Reorder columns: Put Label and Source_File at the start
            cols = ['Label', 'Source_File'] + [c for c in df.columns if c not in ['Label', 'Source_File']]
            df = df[cols]
            
            combined_data.append(df)
            print(f"   Added {fname:<20} ({len(df)} samples) -> {label_str}")

        except Exception as e:
            print(f"   Error reading {filename}: {e}")

    # 4. Concatenate
    if combined_data:
        master_df = pd.concat(combined_data, ignore_index=True)
        
        print("\n" + "="*40)
        print("COMBINATION COMPLETE")
        print(f"Total Samples: {len(master_df)}")
        print(f"Total Columns: {len(master_df.columns)}")
        print("Label Distribution:")
        print(master_df['Label'].value_counts())
        print("="*40)
        
        # 5. Save to the channel directory (where other notebooks expect it)
        output_filename = os.path.join(search_path, "combined_uwb_dataset.csv")
        master_df.to_csv(output_filename, index=False)
        print(f"Saved master dataset to: {output_filename}")
        
        return master_df
    else:
        return None

# Run it
df_combined = combine_all_datasets()