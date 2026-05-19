import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# Data from the user's prompt
methods = ['IDE (ResNet-50)', 'TriNet', 'PCB', 'APR (Paper)', 'Ours (IPAID+IICL)']

# Single-camera data
single_cam_r1 = [65.3, 68.1, 70.2, 72.5, 86.02]
single_cam_map = [52.7, 55.4, 57.8, 60.1, 64.71]

# Cross-camera data
cross_cam_r1 = [47.9, 50.2, 52.1, 55.3, 72.34]
cross_cam_map = [26.5, 28.3, 29.6, 31.8, 37.63]

# Output directory
output_dir = 'outputs/paper_figures'
os.makedirs(output_dir, exist_ok=True)

def plot_grouped_bar(metric_name, single_data, cross_data, filename, y_limit=None):
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors: Blueish for Single-cam, Orangeish for Cross-cam
    rects1 = ax.bar(x - width/2, single_data, width, label='Single-camera', color='#5B9BD5', edgecolor='white')
    rects2 = ax.bar(x + width/2, cross_data, width, label='Cross-camera', color='#ED7D31', edgecolor='white')

    ax.set_ylabel(f'{metric_name} (%)', fontsize=12)
    ax.set_title(f'ATRW Open-Set {metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    if y_limit:
        ax.set_ylim(0, y_limit)
    else:
        ax.set_ylim(0, 100)

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved {filename} to {save_path}")
    plt.close()

# Plot 1: Rank-1 Comparison
plot_grouped_bar('Rank-1', single_cam_r1, cross_cam_r1, 'atrw_openset_rank1_comparison.png')

# Plot 2: mAP Comparison
plot_grouped_bar('mAP', single_cam_map, cross_cam_map, 'atrw_openset_map_comparison.png', y_limit=80)

print("All plots generated successfully.")
