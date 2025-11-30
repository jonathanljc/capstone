import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Set style for publication quality
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# --- CONFIGURATION: YOUR LOCAL PATH ---
base_path = r"C:\Users\preca\dwm1001-examples\new"

# 2. File configuration
files = {
    '1.0m': os.path.join(base_path, '1m_141_home.csv'),
    '1.5m': os.path.join(base_path, '1.5m_141_home.csv'),
    '2.0m': os.path.join(base_path, '2m_141_home.csv')
}

# 3. Define a nice color palette
palette = sns.color_palette("deep", 3)
colors = {'1.0m': palette[0], '1.5m': palette[1], '2.0m': palette[2]}

# 4. Increase figure size significantly (16x9 inches)
plt.figure(figsize=(16, 9))

# DETERMINE MAX LENGTH & MAX VALUE FOR SCALING
max_len = 0
max_val = 0
for filepath in files.values():
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if len(df) > max_len: max_len = len(df)
            if df['Distance'].max() > max_val: max_val = df['Distance'].max()
        except: pass

# Set X-Axis Limit with Padding (The Fix)
plt.xlim(0, max_len * 1.15)
# Set Y-Axis Limit with EXTRA Padding for Legend (The Fix)
plt.ylim(0, max_val * 1.5) # 50% headroom

# 5. Plot Loop
for label, filepath in files.items():
    try:
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found at {filepath}")
            continue

        df = pd.read_csv(filepath)
        mean_dist = df['Distance'].mean()
        true_dist = float(label.replace('m', ''))
        color = colors[label]
        
        # Plot Raw Data as THINNER Pulse Line
        plt.plot(df.index, df['Distance'], label=f'{label} Measured Signal', 
                 linewidth=1.0, alpha=0.6, color=color)
        
        # Plot Mean (Solid & THINNER)
        plt.axhline(y=mean_dist, color=color, linestyle='-', linewidth=2.0, 
                    label=f'{label} Mean ({mean_dist:.3f}m)')
        
        # Plot True Distance (Dashed & THINNER)
        plt.axhline(y=true_dist, color=color, linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'{label} Target ({true_dist:.1f}m)')
        
        # Add text annotation on the right (With White Box)
        plt.text(max_len * 1.02, mean_dist, f"{mean_dist:.3f}m\n(Err: {mean_dist - true_dist:+.3f}m)", 
                 color=color, va='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'))
        
        print(f"✅ Plotting {label}...")
        
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")

# 6. Labels and Title
plt.title('UWB Calibration Validation: 1m, 1.5m, 2m', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Sample Index (Time)', fontsize=16, labelpad=15)
plt.ylabel('Measured Distance (m)', fontsize=16, labelpad=15)

# 7. Customize legend placement (Top Right, Horizontal)
plt.legend(loc='upper right', ncol=3, frameon=True, fontsize=12, 
           framealpha=0.95, borderpad=1, shadow=True)

# 8. Adjust layout to make room for legend
plt.tight_layout()

# 9. Save and Show
plt.savefig(os.path.join(base_path, 'final_calibration_plot_thinner.png'), dpi=300)
print(f"\nGraph saved to: {os.path.join(base_path, 'final_calibration_plot_thinner.png')}")
plt.show()