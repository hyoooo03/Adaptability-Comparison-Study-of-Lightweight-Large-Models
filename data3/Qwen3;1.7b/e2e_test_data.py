import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
CSV_FILE = "e2e_test_data_163243.csv"

# --- 核心配置：添加中文字体 ---
# 确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 加载数据
# ==========================================
try:
    df = pd.read_csv(CSV_FILE,encoding='gbk')
    print(f"✅ 成功加载数据: {CSV_FILE}")
    print(f"📊 数据维度: {df.shape[0]} 行, {df.shape[1]} 列")
except FileNotFoundError:
    print(f"❌ 错误: 未找到文件 {CSV_FILE}")
    exit()

# ==========================================
# 2. 计算核心指标 (按论文公式)
# ==========================================
# 2.1 全链路端到端延迟 (L_e2e)
df['L_E2E'] = (df['T_End'] - df['T_Start']) * 1000  # 转换为毫秒

# 2.2 环节耗时分解
df['L_ASR'] = (df['T_ASR_End'] - df['T_Start']) * 1000
df['L_LLM'] = (df['T_LLM_End'] - df['T_LLM_Start']) * 1000
df['L_TTS'] = (df['T_End'] - df['T_TTS_Start']) * 1000

# 2.3 首字延迟 (TTFT)
df['TTFT'] = (df['T_LLM_End'] - df['T_LLM_Start']) * 1000

# ==========================================
# 3. 统计分析与报告
# ==========================================
print("\n" + "="*50)
print("📊 实验三：端到端语音交互全链路测试分析报告")
print("="*50)

# 3.1 汇总统计
stats = df[['L_E2E', 'L_ASR', 'L_LLM', 'L_TTS', 'TTFT']].describe()

print(f"\n📋 指标汇总统计 (单位: ms):")
print("-" * 80)
print(f"{'指标':<15} {'平均值':<10} {'中位数':<10} {'最小值':<10} {'最大值':<10} {'标准差':<10}")
print("-" * 80)
for col in ['L_E2E', 'L_ASR', 'L_LLM', 'L_TTS', 'TTFT']:
    mean_val = stats.loc['mean', col]
    median_val = stats.loc['50%', col]
    min_val = stats.loc['min', col]
    max_val = stats.loc['max', col]
    std_val = stats.loc['std', col]
    print(f"{col:<15} {mean_val:<10.2f} {median_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f} {std_val:<10.2f}")
print("-" * 80)

# --- 修改点：使用新的区间标准进行统计 ---
print(f"\n📊 端到端延迟响应区间分布统计:")

# 1. 定义区间边界和标签 (单位：毫秒)
# 对应你给的标准: 0-10s, 10-15s, 15-30s, >30s
bins = [0, 10000, 15000, 30000, float('inf')]
labels = ['0-10s (流畅)', '10-15s (一般)', '15-30s (较慢)', '>30s (极慢)']

# 2. 进行分组统计
df['Latency_Bin'] = pd.cut(df['L_E2E'], bins=bins, labels=labels, right=False)
bin_counts = df['Latency_Bin'].value_counts().sort_index()
total_count = len(df)

print("-" * 40)
print(f"{'区间':<15} {'数量':<8} {'占比':<8}")
print("-" * 40)
for label, count in bin_counts.items():
    percentage = (count / total_count) * 100
    print(f"{label:<15} {count:<8} {percentage:<8.1f}%")
print("-" * 40)

# 3.3 保存计算结果
output_file = f"analysis_results_{pd.Timestamp.now().strftime('%H%M%S')}.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 详细计算结果已保存至: {output_file}")

# ==========================================
# 4. 可视化 (用于论文图表)
# ==========================================
# 增大画布高度以容纳20个标签
plt.figure(figsize=(14, 10))

# 4.1 延迟分布箱线图
plt.subplot(2, 1, 1)
sns.boxplot(data=df[['L_E2E', 'L_ASR', 'L_LLM', 'L_TTS']])
plt.title('端到端各环节延迟分布 (Box Plot)')
plt.ylabel('延迟 (ms)')
plt.grid(True, linestyle='--', alpha=0.7)

# 4.2 堆叠柱状图 (展示全部 20 轮)
plt.subplot(2, 1, 2)

NToShow = len(df)  # 使用全部 20 组数据
labels = [f"Round {i+1}" for i in range(NToShow)]
asr_data = df['L_ASR'].values
llm_data = df['L_LLM'].values
tts_data = df['L_TTS'].values

index = np.arange(NToShow)

# 增加 bar_width 让柱子稍微宽一点，更美观
bar_width = 0.8
plt.bar(index, asr_data, bar_width, label='ASR 耗时', alpha=0.8)
plt.bar(index, llm_data, bar_width, bottom=asr_data, label='LLM 耗时', alpha=0.8)
plt.bar(index, tts_data, bar_width, bottom=asr_data+llm_data, label='TTS 耗时', alpha=0.8)

plt.xlabel('     ')
plt.ylabel('延迟 (ms)')
plt.title(f'全 {NToShow} 轮全链路延迟成分拆解 (Stacked Bar)')
plt.xticks(index, labels, rotation=90, fontsize=9)  # 垂直显示标签
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 调整图例位置防止遮挡
plt.tight_layout()

plt.show()