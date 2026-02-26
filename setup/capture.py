import pandas as pd

LOG_FILE = r'C:\Users\preca\cir_data\putty.log'
OUTPUT_FILE = r'C:\Users\preca\cir_data\10.19m_los_c1.csv'

print(f"Reading {LOG_FILE}...")

with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

all_data = []
capture_id = 0

for line in lines:
    if line.startswith('DATA|'):
        try:
            # Remove leading "DATA|" and trailing newline or "|"
            parts = line[5:].strip().rstrip('|').split('|')

            # Need at least metadata fields
            if len(parts) < 8:
                continue

            # Metadata
            row = {
                'Capture_ID': capture_id,
                'Distance': float(parts[0]),
                'FP_INDEX': int(parts[1]),
                'FP_AMPL1': int(parts[2]),
                'FP_AMPL2': int(parts[3]),
                'FP_AMPL3': int(parts[4]),
                'RXPACC': int(parts[5]),
                'STD_NOISE': int(parts[6]),
                'MAX_NOISE': int(parts[7]),
            }

            # CIR data only magnitude now
            cir_start = 8
            sample_num = 0

            for val in parts[cir_start:]:
                try:
                    row[f'CIR{sample_num}'] = int(val)
                    sample_num += 1
                except:
                    # ignore bad values
                    pass

            if sample_num >= 1000:
                all_data.append(row)
                print(f"✓ Capture {capture_id}  {sample_num} CIR samples")
                capture_id += 1
            else:
                print(f"✗ Capture {capture_id}  only {sample_num} samples")

        except Exception as e:
            print(f"✗ Error  {e}")

if len(all_data) > 0:
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*60)
    print("Success  Magnitude only dataset created")
    print("="*60)
    print(f"Total captures  {len(df)}")
    print(f"Total columns  {len(df.columns)}")
    print(f"  Metadata  9")
    print(f"  CIR Magnitude  {len(df.columns) - 9}")
    print("="*60 + "\n")

    print("Preview")
    print(df.head())
else:
    print("\n✗ No valid data found")
