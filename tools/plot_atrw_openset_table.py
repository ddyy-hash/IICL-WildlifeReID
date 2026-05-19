import matplotlib.pyplot as plt
import pandas as pd
import os

# Data
data = {
    "Method": ["APR (ATRW 2020)", "Ours (IPAID+IICL)", "Improvement"],
    "Single-cam Rank-1": ["72.5%", "86.02%", "+13.5%"],
    "Single-cam mAP": ["60.1%", "64.71%", "+4.6%"],
    "Cross-cam Rank-1": ["55.3%", "72.34%", "+17.0%"],
    "Cross-cam mAP": ["31.8%", "37.63%", "+5.8%"]
}

df = pd.DataFrame(data)

# Output directory
output_dir = 'outputs/paper_figures'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'atrw_openset_table.png')

# Plotting
fig, ax = plt.subplots(figsize=(10, 3)) # Adjust size as needed
ax.axis('off')

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

# Styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5) # Scale width and height

# Header styling
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e') # Dark blue header
        cell.set_edgecolor('white')
    else:
        cell.set_edgecolor('#dddddd')
        if row == 3: # Improvement row
             cell.set_text_props(weight='bold', color='#d62728') # Red for improvement
             cell.set_facecolor('#f9f9f9')
        elif row == 2: # Ours row
             cell.set_text_props(weight='bold')
             cell.set_facecolor('#e6f3ff') # Light blue for Ours
        else:
             cell.set_facecolor('white')

# Adjust column widths specifically if needed, or rely on auto
# Let's make the Method column a bit wider if needed, but auto usually works well for this size.

plt.title("ATRW Open-Set Evaluation Results", pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved table to {save_path}")
