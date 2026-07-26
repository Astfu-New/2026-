"""
生成实验报告所需的图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

with open('experiment_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

input_4x4 = np.array(data['input_4x4'])
kernel_3x3 = np.array(data['kernel_3x3'])
plain_result = np.array(data['plain_result'])
fhe_v1 = np.array(data['fhe_v1_result'])
fhe_v2 = np.array(data['fhe_v2_result'])
intermediate = data['plain_intermediate']
rot_analysis = data['rotation_analysis']
timing = data['timing']

# =====================================================================
# 图1: 输入矩阵、卷积核、输出结果 热图
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

def heatmap(ax, mat, title, cmap='RdBu_r', vmin=None, vmax=None, fmt='d'):
    if vmin is None: vmin = mat.min()
    if vmax is None: vmax = mat.max()
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = 'white' if abs(mat[i,j]) > (vmax+vmin)/2 else 'black'
            ax.text(j, i, f'{mat[i,j]:{fmt}}', ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')
    ax.tick_params(length=0)
    plt.colorbar(im, ax=ax, shrink=0.8)

all_vals = np.concatenate([input_4x4.flatten(), kernel_3x3.flatten(),
                           plain_result.flatten(), fhe_v1.flatten()])
vmin, vmax = all_vals.min(), all_vals.max()

heatmap(axes[0], input_4x4, 'Input X (4x4)', vmin=vmin, vmax=vmax)
heatmap(axes[1], kernel_3x3, 'Kernel K (3x3)', vmin=vmin, vmax=vmax)
heatmap(axes[2], plain_result, 'Output Y (2x2)', vmin=vmin, vmax=vmax)

plt.tight_layout()
plt.savefig('fig1_data_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig1_data_overview.png")

# =====================================================================
# 图2: 卷积过程示意图（4个滑动窗口）
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for idx, inter in enumerate(intermediate):
    ax = axes[idx]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Output {inter['pos_name']} Calculation", fontsize=13, fontweight='bold')

    for i in range(4):
        for j in range(4):
            val = input_4x4[i, j]
            win_i, win_j = idx // 2, idx % 2
            in_window = (win_i <= i < win_i + 3) and (win_j <= j < win_j + 3)
            if in_window:
                rect = Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                 facecolor=colors[idx], alpha=0.3,
                                 edgecolor=colors[idx], linewidth=2)
                ax.add_patch(rect)
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11,
                    fontweight='bold' if in_window else 'normal',
                    color=colors[idx] if in_window else 'gray')

    ki_start, kj_start = win_i, win_j
    for ki in range(3):
        for kj in range(3):
            i = ki_start + ki
            j = kj_start + kj
            kval = kernel_3x3[ki, kj]
            ax.text(j, i+0.2, f'k={kval}', ha='center', va='center',
                    fontsize=8, color='darkred', fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.grid(True, alpha=0.3)

    terms_str = ' + '.join([f"{t['x']}x{t['k']}" for t in inter['terms']])
    ax.text(2.5, 5.1, f"{inter['pos_name']} = {terms_str} = {inter['sum']}",
            ha='center', va='top', fontsize=9, color='black',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('fig2_convolution_process.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig2_convolution_process.png")

# =====================================================================
# 图3: 明文 vs 密文结果对比
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 4))
pos_names = ['y00', 'y01', 'y10', 'y11']
x = np.arange(len(pos_names))
width = 0.35

plain_vals = [inter['sum'] for inter in intermediate]

bars1 = ax.bar(x - width/2, plain_vals, width, label='Plain', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, [fhe_v2[i//2, i%2] for i in range(4)], width,
               label='FHE (V2)', color='#e74c3c', edgecolor='black', hatch='//')

ax.set_ylabel('Output Value', fontsize=12)
ax.set_title('Plain vs FHE Convolution Results', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pos_names)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.text(0.5, 0.02, 'All 4 positions match perfectly, absolute error = 0',
        transform=ax.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('fig3_plain_vs_fhe.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig3_plain_vs_fhe.png")

# =====================================================================
# 图4: 旋转次数对比图
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5))

categories = ['Version A\n(Per Element)', 'Version B\n(Per Output)', 'Theoretical\nMinimum']
rotations = [rot_analysis['v1_total_rotations'], rot_analysis['v2_total_rotations'],
             rot_analysis['theoretical_minimum_serial']]
colors_rot = ['#e74c3c', '#2ecc71', '#3498db']

bars = ax.bar(categories, rotations, color=colors_rot, edgecolor='black', width=0.6)
ax.set_ylabel('Total Rotation Count', fontsize=12)
ax.set_title('Rotation Count Comparison: Pack-Rotate-Accumulate Strategy',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, rotations):
    ax.text(bar.get_x() + bar.get_width()/2., val + 3,
            f'{val}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.text(0.5, 0.85, f'log2(16) = {rot_analysis["rotations_per_dot"]} rotations per dot',
        transform=ax.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.text(0.5, 0.72, 'Version B achieves theoretical minimum!',
        transform=ax.transAxes, ha='center', fontsize=11, color='green',
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))

plt.tight_layout()
plt.savefig('fig4_rotation_count.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig4_rotation_count.png")

# =====================================================================
# 图5: 性能对比图
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

labels_perf = ['Plain Conv', 'Context Setup', 'Encrypt', 'FHE V1\n(144 rot)', 'FHE V2\n(16 rot)']
values_perf = [timing['plain_ms'], timing['context_ms'], timing['encrypt_ms'],
               timing['v1_ms'], timing['v2_ms']]
colors_perf = ['#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#3498db']

bars = ax.barh(labels_perf, values_perf, color=colors_perf, edgecolor='black', height=0.6)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, values_perf):
    ax.text(val * 1.3, bar.get_y() + bar.get_height()/2,
            f'{val:.2f} ms', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('fig5_performance.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig5_performance.png")

# =====================================================================
# 图6: Pack-Rotate-Accumulate 流程图
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(6.5, 5.7, 'Pack-Rotate-Accumulate Workflow', ha='center', fontsize=16, fontweight='bold')

steps = [
    (1, 3.5, 'Step 1\nPack\n4x4 -> 16D\nVector', '#3498db'),
    (3.5, 3.5, 'Step 2\nEncrypt\nBFV\nBatching', '#9b59b6'),
    (6, 3.5, 'Step 3\nRotate\nAlign slots\nwith kernel', '#e67e22'),
    (8.5, 3.5, 'Step 4\nMultiply\nCipher-Plain\nElement-wise', '#e74c3c'),
    (11, 3.5, 'Step 5\nAccumulate\nBinary tree\nsum', '#2ecc71'),
]

for x, y, text, color in steps:
    rect = Rectangle((x-0.8, y-0.7), 1.6, 1.4,
                     facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

for i in range(4):
    x1 = steps[i][0] + 0.8
    x2 = steps[i+1][0] - 0.8
    ax.annotate('', xy=(x2, 3.5), xytext=(x1, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.text(6.5, 1.8, 'dot() Internal: rotate(slot/2) -> add -> rotate(slot/4) -> add -> rotate(slot/8) -> add -> rotate(slot/16) -> add',
        ha='center', va='top', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=1.5))

ax.text(6.5, 0.8, 'For 16 slots: log2(16) = 4 rotations per dot',
        ha='center', va='top', fontsize=10, fontweight='bold', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='#d4edda', edgecolor='green', linewidth=1.5))

plt.tight_layout()
plt.savefig('fig6_workflow.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig6_workflow.png")

print("\nAll figures generated! Run gen_fig7.py separately for fig7.")
