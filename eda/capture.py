import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup the File List ---
# Format: (Filename, True_Distance_in_Meters, Label_Category)
files_config = [
    ('LOS_2m_living_room_home.csv',              2.0, 'LOS'),
    ('LOS_4.3m_living_room_corner_home.csv',     4.3, 'LOS'),
    ('NLOS_2.6m_open_door_home.csv',             2.6, 'NLOS'),
    ('NLOS_4.4m_close_door_home.csv',            4.4, 'NLOS')
]

data_frames = []

# --- 2. Process Each File ---
print("Processing Data Files...")
for filename, true_dist, category in files_config:
    try:
        # Read CSV. We let pandas detect the header automatically.
        # We only need the 'Distance' column.
        df = pd.read_csv(filename, usecols=['Distance'])
        
        # Calculate Error: (Measured - True)
        # Positive Error = UWB thinks it's further away (common in NLOS)
        df['Error'] = df['Distance'] - true_dist
        
        # Tag the data
        df['Category'] = category
        df['Scenario'] = f"{category} ({true_dist}m)"
        
        data_frames.append(df)
        print(f"  -> Loaded {filename}: {len(df)} samples. Mean Error: {df['Error'].mean():.3f}m")
        
    except Exception as e:
        print(f"  x Error reading {filename}: {e}")

# Combine into one big dataset
if not data_frames:
    print("No data loaded!")
    exit()

all_data = pd.concat(data_frames, ignore_index=True)

# --- 3. Print Statistics for your Report ---
print("\n=== RANGING ERROR STATISTICS (meters) ===")
stats = all_data.groupby('Category')['Error'].describe()[['count', 'mean', 'std', 'min', 'max']]
print(stats)
print("=========================================\n")

# --- 4. Generate the Box Plot ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# Create Box Plot
ax = sns.boxplot(x="Category", y="Error", hue="Category", data=all_data, 
                 palette={"LOS": "#2ecc71", "NLOS": "#e74c3c"}, legend=False)

# Formatting
plt.title('Ranging Error Distribution: LOS vs. NLOS', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Propagation Condition', fontsize=12)
plt.ylabel('Ranging Error (meters)', fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7) # Zero error line

# Add Mean Labels on the plot
means = all_data.groupby(['Category'])['Error'].mean()
for i, cat in enumerate(['LOS', 'NLOS']):
    plt.text(i, means[cat] + 0.05, f"Mean: {means[cat]:+.2f}m", 
             horizontalalignment='center', fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('Ranging_Error_Boxplot.png', dpi=300)
plt.show()

print("Graph saved as 'Ranging_Error_Boxplot.png'")