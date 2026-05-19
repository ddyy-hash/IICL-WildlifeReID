import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

output_dir = 'outputs/paper_figures'
os.makedirs(output_dir, exist_ok=True)

def save_table_image(df, title, filename):
    # Calculate figure size based on rows and columns
    fig_width = len(df.columns) * 2.5
    fig_height = len(df) * 0.5 + 1.5
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6) # Increase row height
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # Dark blue header
            cell.set_edgecolor('white')
        else:
            cell.set_edgecolor('#dddddd')
            row_data = df.iloc[row-1]
            method_name = str(row_data[0])
            
            # Highlight 'Ours' row
            if 'Ours' in method_name:
                cell.set_facecolor('#e6f3ff') # Light blue
                cell.set_text_props(weight='bold')
            elif 'Improvement' in method_name:
                cell.set_facecolor('#fff5f5') # Light red
                cell.set_text_props(weight='bold', color='#d62728')
            else:
                cell.set_facecolor('white')

    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {filename} to {save_path}")
    plt.close()

# ==========================================
# Table 1: Closed-Set Comparison
# ==========================================
data_closed = {
    "Method": ["ResNet50", "PCB", "SMFFEN 2024", "Ours (IPAID+IICL)"],
    "Rank-1": ["91.70%", "94.70%", "96.30%", "97.88%"],
    "Rank-5": ["97.90%", "98.40%", "98.90%", "99.35%"],
    "mAP": ["68.40%", "71.20%", "78.70%", "79.15%"]
}
df_closed = pd.DataFrame(data_closed)
save_table_image(df_closed, "ATRW Closed-Set Comparison (7:3 Split)", "atrw_closedset_table.png")

# ==========================================
# Table 2: Open-Set Comparison
# ==========================================
data_open = {
    "Method": ["IDE (ResNet-50)", "TriNet", "PCB", "APR (ATRW 2020)", "Ours (IPAID+IICL)", "Improvement (vs APR)"],
    "Single-cam Rank-1": ["65.3%", "68.1%", "70.2%", "72.5%", "86.02%", "+13.5%"],
    "Single-cam mAP": ["52.7%", "55.4%", "57.8%", "60.1%", "64.71%", "+4.6%"],
    "Cross-cam Rank-1": ["47.9%", "50.2%", "52.1%", "55.3%", "72.34%", "+17.0%"],
    "Cross-cam mAP": ["26.5%", "28.3%", "29.6%", "31.8%", "37.63%", "+5.8%"]
}
df_open = pd.DataFrame(data_open)
save_table_image(df_open, "ATRW Open-Set Comparison (Official Protocol)", "atrw_openset_comparison_table.png")
