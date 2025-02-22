import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# 数据保持不变
pareto_values = [
    [0.1814348398710112, 0.061159348487854],
    [0.18780197164876927, 0.06517689704895019],
    [0.18996159222159184, 0.06706037521362304],
    [0.1943714418337548, 0.06862882614135742],
    [0.2009580635259764, 0.07115460395812988],
    [0.20133776820696914, 0.07368038177490234],
    [0.20891041783435846, 0.07727590560913086]
]

arch_configs = [
    {'ks': [5,3,7,3,5,5,5,3,3,3,5,7,3,3,5,7,3,5,5,7], 'e': [6,6,6,3,4,6,4,3,4,6,4,3,6,4,6,4,6,4,4,4], 'd': [2,2,2,3,2]},
    {'ks': [7,3,7,5,3,7,7,7,5,7,3,3,5,3,3,7,5,7,3,5], 'e': [4,3,3,6,3,6,6,4,6,6,3,4,6,6,3,6,4,6,3,4], 'd': [4,2,2,2,2]},
    {'ks': [7,3,5,7,5,7,5,7,7,7,5,3,5,5,5,7,7,5,7,5], 'e': [6,4,4,6,4,6,4,6,6,3,3,4,6,6,4,3,4,3,3,6], 'd': [3,2,2,3,2]},
    {'ks': [3,3,3,3,5,7,3,7,3,7,7,7,3,7,7,5,3,5,3,3], 'e': [6,4,6,6,6,4,6,4,6,6,6,6,6,6,6,3,6,3,6,3], 'd': [3,4,2,2,3]},
    {'ks': [5,3,5,5,5,3,3,5,7,3,3,5,3,3,7,3,3,7,5,3], 'e': [6,6,4,4,4,6,3,4,6,4,3,6,3,3,3,4,6,6,4,6], 'd': [3,2,4,3,3]},
    {'ks': [3,5,7,3,3,7,3,7,7,3,3,7,5,7,3,3,5,3,5,7], 'e': [3,4,3,6,4,6,3,6,4,3,4,6,6,6,4,3,4,4,6,4], 'd': [3,4,3,2,2]},
    {'ks': [3,3,7,3,5,5,3,7,5,5,7,7,5,3,3,7,7,3,3,7], 'e': [3,4,3,3,3,6,3,4,4,6,3,3,4,4,4,3,4,3,3,4], 'd': [4,3,3,3,3]}
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# 创建画布和布局
fig = plt.figure(figsize=(16, 14))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

# 帕累托图
ax_pareto = fig.add_subplot(gs[0])

for i, (acc, lat) in enumerate(pareto_values):
    ax_pareto.scatter(lat, acc, color=colors[i], s=150, edgecolor='white', linewidth=2, zorder=5, label=f'Arch {i+1}')
    ax_pareto.text(lat, acc + 0.001, f'Arch {i+1}', fontsize=12, ha='center', va='bottom')

ax_pareto.plot([v[1] for v in pareto_values], [v[0] for v in pareto_values], 
        '--', color='gray', alpha=0.7, linewidth=2, label='Pareto Front')

ax_pareto.set_xlabel('Latency (lower is better)', fontsize=14, labelpad=10)
ax_pareto.set_ylabel('Accuracy (higher is better)', fontsize=14, labelpad=10)
ax_pareto.grid(True, linestyle='--', alpha=0.3)
ax_pareto.set_title('Pareto Front of Neural Architectures', fontsize=16, pad=20)
ax_pareto.tick_params(axis='both', which='major', labelsize=12)

# 表格
ax_table = fig.add_subplot(gs[1])
ax_table.axis('off')

# 准备表格数据
table_data = []
# 添加表头
table_data.append(['Architecture', 'Kernel Sizes (ks)', 'Expansion Ratios (e)', 'Depth (d)'])

# 添加每个架构的数据
for i, config in enumerate(arch_configs):
    ks_str = f"[{', '.join(map(str, config['ks']))}]"
    e_str = f"[{', '.join(map(str, config['e']))}]"
    d_str = f"[{', '.join(map(str, config['d']))}]"
    table_data.append([f'Arch {i+1}', ks_str, e_str, d_str])

# 创建表格
table = ax_table.table(cellText=table_data,
                      loc='center',
                      cellLoc='center',
                      edges='closed',
                      colWidths=[0.1, 0.35, 0.35, 0.2])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)  # 调整表格大小

# 设置表头样式
for i in range(len(table_data[0])):
    table[(0, i)].set_text_props(weight='bold')
    table[(0, i)].set_facecolor('#f0f0f0')

# 为每行设置对应的颜色
for i in range(1, len(table_data)):
    color = colors[i-1]
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if j == 0:  # Architecture 列
            cell.set_text_props(weight='bold', color=color)
        else:  # 其他列
            cell.set_text_props(color=color)

plt.tight_layout()
plt.show()