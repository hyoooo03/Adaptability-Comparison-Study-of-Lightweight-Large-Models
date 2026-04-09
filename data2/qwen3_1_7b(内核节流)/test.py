import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =================配置区域=================
TRACE_FILE = 'raw_thermal_trace_20260323_000729.csv'
LOG_FILE = 'raw_inference_log_20260323_000729.csv'
OUTPUT_PREFIX = 'scene2_analysis'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =================1. 数据加载与清洗=================
print("正在加载数据...")

# --- 修正点 1: 尝试自动识别表头，并强制转换数据类型 ---
try:
    # 先尝试读取前几行看看是否有表头
    test_df = pd.read_csv(TRACE_FILE, nrows=2)
    # 如果第一行第一列是字符串且看起来像列名（不是数字），则认为有表头
    has_header = not test_df.iloc[0, 0].replace('.', '', 1).isdigit()
except:
    has_header = False

if has_header:
    print("检测到文件包含表头，自动使用文件自带列名...")
    df_trace = pd.read_csv(TRACE_FILE)
    # 重命名可能的英文列名为统一格式（以防万一）
    # 假设原表头可能是 Time, Temp, Freq, Mem, Throttle 等
    # 这里做一个简单的映射检查，如果列名不匹配，手动指定
    expected_cols = ['Elapsed_Seconds', 'Temp_C', 'Freq_MHz', 'Mem_Used_MB', 'Throttle_Flag']
    if list(df_trace.columns) != expected_cols:
        # 如果列名不对，尝试按位置重命名（假设顺序一致）
        df_trace.columns = expected_cols
else:
    print("未检测到表头，使用默认列名...")
    trace_cols = ['Elapsed_Seconds', 'Temp_C', 'Freq_MHz', 'Mem_Used_MB', 'Throttle_Flag']
    df_trace = pd.read_csv(TRACE_FILE, header=None, names=trace_cols)

# --- 修正点 2: 强制将关键列转换为数值类型 ---
numeric_cols = ['Elapsed_Seconds', 'Temp_C', 'Freq_MHz', 'Mem_Used_MB', 'Throttle_Flag']
for col in numeric_cols:
    if col in df_trace.columns:
        # errors='coerce' 会将无法转换的字符变为 NaN，避免报错
        df_trace[col] = pd.to_numeric(df_trace[col], errors='coerce')

# 删除因转换失败产生的 NaN 行（通常是表头行或脏数据）
df_trace = df_trace.dropna(subset=['Elapsed_Seconds'])

# 1.2 加载 Inference Log 数据
df_log = pd.read_csv(LOG_FILE)

# 解析 Log 中的时间
start_time_str = df_log['Start_Time_ISO'].iloc[0]
start_time = datetime.fromisoformat(start_time_str)

def get_elapsed_seconds(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        return (dt - start_time).total_seconds()
    except:
        return 0

df_log['Start_Sec'] = df_log['Start_Time_ISO'].apply(get_elapsed_seconds)
df_log['End_Sec'] = df_log['Start_Sec'] + df_log['Duration_s']

print(f"Trace 数据有效行数: {len(df_trace)}")
print(f"Log 数据任务数: {len(df_log)}")
print(f"实验总时长: {df_trace['Elapsed_Seconds'].max():.1f} 秒")

# =================2. 构建 TPS 时间序列=================
df_trace['TPS'] = np.nan

for _, row in df_log.iterrows():
    t_start = row['Start_Sec']
    t_end = row['End_Sec']
    tps_val = row['TPS']
    
    mask = (df_trace['Elapsed_Seconds'] >= t_start) & (df_trace['Elapsed_Seconds'] <= t_end)
    df_trace.loc[mask, 'TPS'] = tps_val

df_trace['TPS'] = df_trace['TPS'].ffill()

# =================3. 计算核心指标=================
print("\n正在计算核心指标...")

freq_threshold = 2350
temp_threshold = 80.0

throttling_rows = df_trace[(df_trace['Freq_MHz'] < freq_threshold) | (df_trace['Temp_C'] > temp_threshold)]

if not throttling_rows.empty:
    t_golden_idx = throttling_rows.index[0]
    t_golden = df_trace.loc[t_golden_idx, 'Elapsed_Seconds']
else:
    t_golden = df_trace['Elapsed_Seconds'].max()

initial_mask = df_trace['Elapsed_Seconds'] <= min(30, t_golden)
tps_initial = df_trace.loc[initial_mask, 'TPS'].mean()

steady_mask = df_trace['Elapsed_Seconds'] >= (df_trace['Elapsed_Seconds'].max() - 60)
tps_steady = df_trace.loc[steady_mask, 'TPS'].mean()

if tps_initial > 0:
    eta_deg = (tps_initial - tps_steady) / tps_initial * 100
else:
    eta_deg = 0

temp_steady = df_trace.loc[steady_mask, 'Temp_C'].mean()
freq_steady = df_trace.loc[steady_mask, 'Freq_MHz'].mean()

degrade_mask = (df_trace['Temp_C'] > 80) & (df_trace['Freq_MHz'] < 2400)
slope_k = 0
if degrade_mask.sum() > 10:
    deg_data = df_trace[degrade_mask]
    k_coeff = np.polyfit(deg_data['Temp_C'], deg_data['Freq_MHz'], 1)
    slope_k = k_coeff[0]

print(f"--- 场景二核心指标 ---")
print(f"1. 满血窗口期 (T_golden): {t_golden:.2f} s")
print(f"2. 初始 TPS: {tps_initial:.2f} tokens/s")
print(f"3. 稳态 TPS: {tps_steady:.2f} tokens/s")
print(f"4. 热致性能衰减率: {eta_deg:.2f}%")
print(f"5. 稳态温度/频率: {temp_steady:.1f}°C / {freq_steady:.1f} MHz")
print(f"6. 温度 - 频率耦合斜率: {slope_k:.2f} MHz/°C")

# =================4. 绘图=================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 1]})

c_temp = '#d62728'
c_freq = '#1f77b4'
c_mem = '#9467bd'
c_tps = '#2ca02c'

# 子图 1
ax1.plot(df_trace['Elapsed_Seconds'], df_trace['Temp_C'], color=c_temp, linewidth=2, label='CPU 温度 (°C)', zorder=3)
ax1.set_ylabel('温度 (°C)', fontsize=12, color=c_temp, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=c_temp)
ax1.set_ylim(30, 100)

ax1_freq = ax1.twinx()
ax1_freq.step(df_trace['Elapsed_Seconds'], df_trace['Freq_MHz'], where='post', color=c_freq, linewidth=2, label='CPU 频率 (MHz)', alpha=0.8, zorder=2)
ax1_freq.set_ylabel('频率 (MHz)', color=c_freq, fontsize=12, fontweight='bold')
ax1_freq.tick_params(axis='y', labelcolor=c_freq)
ax1_freq.set_ylim(1400, 2500)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_freq.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.set_title(f'Qwen3 : 1.7b    热演化与性能衰减过程 (T_golden = {t_golden:.1f}s)', fontsize=14, pad=10)

ax1.axvline(x=t_golden, color='black', linestyle='--', linewidth=1.5, zorder=10)
ax1.text(t_golden + 10, 95, f'$T_{{golden}}$\n{t_golden:.1f}s', fontsize=11, verticalalignment='top', rotation=90, fontweight='bold', color='black')

# 子图 2
mem_initial = df_trace['Mem_Used_MB'].iloc[0]
df_trace['Mem_Delta'] = df_trace['Mem_Used_MB'] - mem_initial

ax2.plot(df_trace['Elapsed_Seconds'], df_trace['Mem_Delta'], color=c_mem, linewidth=2, label='内存增量 (MB)', zorder=3)
ax2.set_ylabel('内存增量 (MB)', fontsize=12, color=c_mem, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=c_mem)

ax2_tps = ax2.twinx()
ax2_tps.step(df_trace['Elapsed_Seconds'], df_trace['TPS'], where='post', color=c_tps, linewidth=2.5, label='吞吐量 (TPS)', alpha=0.9, zorder=4)
ax2_tps.set_ylabel('生成速度 (Tokens/s)', color=c_tps, fontsize=12, fontweight='bold')
ax2_tps.tick_params(axis='y', labelcolor=c_tps)
ax2_tps.set_ylim(0, max(df_trace['TPS'].max() * 1.2, 5)) # 防止最大值为0报错

#ax2_tps.text(df_trace['Elapsed_Seconds'].max() * 0.05, tps_initial * 0.9, f'初始：{tps_initial:.2f}', color=c_tps, fontsize=10, fontweight='bold', #bbox=dict(facecolor='white', alpha=0.7))
#ax2_tps.text(df_trace['Elapsed_Seconds'].max() * 0.6, tps_steady * 1.1, f'稳态：{tps_steady:.2f}', color=c_tps, fontsize=10, fontweight='bold', #bbox=dict(facecolor='white', alpha=0.7))

ax2.grid(True, linestyle=':', alpha=0.3)
lines_m, labels_m = ax2.get_legend_handles_labels()
lines_t, labels_t = ax2_tps.get_legend_handles_labels()
ax2.legend(lines_m + lines_t, labels_m + labels_t, loc='upper left', fontsize=11, framealpha=0.9)

# 子图 3
ax3.fill_between(df_trace['Elapsed_Seconds'], 0, 1, 
                 where=(df_trace['Throttle_Flag'] == 1), 
                 color='#ff9999', alpha=0.6, label='节流状态')
ax3.fill_between(df_trace['Elapsed_Seconds'], 0, 1, 
                 where=(df_trace['Throttle_Flag'] == 0), 
                 color='#99ff99', alpha=0.3, label='满血状态')

ax3.set_yticks([]) 
ax3.set_xlabel('运行时间 (秒)', fontsize=12)
ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax3.set_xlim(0, df_trace['Elapsed_Seconds'].max())

plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}_timeseries.png', dpi=300)
print(f"\n已保存时序图：{OUTPUT_PREFIX}_timeseries.png")

# 图 2
fig2, ax_phase = plt.subplots(figsize=(10, 8))
scatter = ax_phase.scatter(df_trace['Temp_C'], df_trace['Freq_MHz'], 
                           c=df_trace['TPS'], cmap='viridis', 
                           s=20, alpha=0.6, edgecolors='none')

if slope_k != 0:
    deg_data = df_trace[(df_trace['Temp_C'] > 80) & (df_trace['Freq_MHz'] < 2400)]
    if len(deg_data) > 10:
        z = np.polyfit(deg_data['Temp_C'], deg_data['Freq_MHz'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(deg_data['Temp_C'].min(), deg_data['Temp_C'].max(), 100)
        ax_phase.plot(x_range, p(x_range), 'r--', linewidth=2, label=f'拟合斜率 k={z[0]:.2f}')

ax_phase.set_xlabel('CPU 温度 (°C)', fontsize=12, fontweight='bold')
ax_phase.set_ylabel('CPU 频率 (MHz)', fontsize=12, fontweight='bold')
ax_phase.set_title('Qwen3:1.7b    温度 - 频率耦合关系相平面', fontsize=14)
ax_phase.grid(True, linestyle=':', alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('吞吐量 (TPS)', fontsize=12)
ax_phase.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}_phase_plot.png', dpi=300)
print(f"已保存相平面图：{OUTPUT_PREFIX}_phase_plot.png")

plt.show()