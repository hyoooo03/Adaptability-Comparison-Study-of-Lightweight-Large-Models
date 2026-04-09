import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
df = pd.read_csv('raw_thermal_trace_20260323_000729.csv')

# --- 数据预处理：降采样平滑 + 内存占用计算 ---
df['Second'] = df['Elapsed_Seconds'].astype(int)

# 计算内存占用（减去初始值）
initial_mem = df['Mem_Used_MB'].iloc[0]
df['Mem_Delta_MB'] = df['Mem_Used_MB'] - initial_mem

plot_df = df.groupby('Second').agg({
    'Temp_C': 'mean',
    'Freq_MHz': 'mean',
    'Mem_Delta_MB': 'mean',   # 使用增量
    'Throttle_Flag': 'max'
}).reset_index()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布：3个子图，共享X轴，高度比例 3:1:1
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

# 颜色定义
c_temp = '#d62728' # 红
c_freq = '#1f77b4' # 蓝
c_mem = '#9467bd'  # 紫

# --- 【图1】温度 vs 频率 ---
ax1.plot(plot_df['Second'], plot_df['Temp_C'], color=c_temp, linewidth=2.5, label='CPU 温度 (°C)', zorder=3)
ax1.set_ylabel('温度 (°C)', fontsize=12, color=c_temp, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=c_temp)
ax1.set_ylim(30, 100)

ax1_freq = ax1.twinx()
ax1_freq.step(plot_df['Second'], plot_df['Freq_MHz'], where='post', color=c_freq, linewidth=2, label='CPU 频率 (MHz)', alpha=0.8, zorder=2)
ax1_freq.set_ylabel('频率 (MHz)', color=c_freq, fontsize=12, fontweight='bold')
ax1_freq.tick_params(axis='y', labelcolor=c_freq)
ax1_freq.set_ylim(1400, 2500)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_freq.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.set_title('场景二：热演化物理过程 (温度 vs 频率 vs 内存占用--qwen3:1.7b)', fontsize=14, pad=10)

# --- 【图2】内存占用 (独立坐标轴) ---
ax2.plot(plot_df['Second'], plot_df['Mem_Delta_MB'], color=c_mem, linewidth=2, label='内存占用 (MB)', zorder=3)
ax2.set_ylabel('内存占用 (MB)', fontsize=12, color=c_mem, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=c_mem)

# 动态设置 Y 轴范围：从 0 到最大增量的 1.1 倍
max_delta = plot_df['Mem_Delta_MB'].max()
ax2.set_ylim(0, max_delta * 1.1)  # 留点顶部空间

ax2.grid(True, linestyle=':', alpha=0.3)
ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

# --- 【图3】节流状态 ---
throttle_start_row = plot_df[plot_df['Throttle_Flag'] == 1]
t_golden_val = 0

if not throttle_start_row.empty:
    t_golden_val = throttle_start_row.index[0]
    
    ax3.fill_between(plot_df['Second'], 0, 1, 
                     where=(plot_df['Throttle_Flag'] == 1), 
                     color='#ff9999', alpha=0.6, label='节流状态 (Throttling)')
    
    ax3.fill_between(plot_df['Second'], 0, 1, 
                     where=(plot_df['Throttle_Flag'] == 0), 
                     color='#99ff99', alpha=0.3, label='满血状态 (Golden)')
    
    ax3.axvline(x=t_golden_val, color='black', linestyle='--', linewidth=1.5, zorder=10)
    ax3.text(t_golden_val + 10, 0.6, f'$T_{{golden}}$\n{t_golden_val}s', 
             fontsize=11, verticalalignment='center', rotation=90, fontweight='bold')

ax3.set_yticks([]) 
ax3.set_xlabel('运行时间 (秒)', fontsize=12)
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.set_xlim(0, plot_df['Second'].max())

plt.tight_layout()
plt.show()