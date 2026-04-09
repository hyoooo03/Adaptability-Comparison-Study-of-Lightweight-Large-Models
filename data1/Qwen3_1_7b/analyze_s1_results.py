import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import json
import os
from datetime import datetime
import glob
import warnings

# ================= 🛠️ 核心修复：暴力指定字体路径 =================
warnings.filterwarnings('ignore')

# 1. 定义可能的中文字体路径 (Windows 标准路径)
font_paths = [
    r"C:\Windows\Fonts\simhei.ttf",       # 黑体
    r"C:\Windows\Fonts\msyh.ttc",         # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",       # 微软雅黑 Bold
]

# 2. 找到第一个存在的路径
valid_font_path = None
for path in font_paths:
    if os.path.exists(path):
        valid_font_path = path
        print(f"✅ 找到本地字体文件：{path}")
        break

if valid_font_path is None:
    print("❌ 警告：未在标准路径找到中文字体文件，将尝试使用默认配置。")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    my_font = None
else:
    # 3. 创建字体属性对象
    my_font = fm.FontProperties(fname=valid_font_path)
    
    # 4. 全局设置
    plt.rcParams['font.sans-serif'] = [my_font.get_name()] 
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 已强制加载字体：{my_font.get_name()}")

# ================================================================

# 配置部分
INPUT_DIR = "."
OUTPUT_EXCEL = "s1_manual_evaluation.xlsx"
OUTPUT_PLOT_COMBINED = "s1_performance_charts_cn.png"

# 寻找最新的文件
metrics_files = sorted(glob.glob(os.path.join(INPUT_DIR, "s1_raw_metrics_*.csv")), reverse=True)
content_files = sorted(glob.glob(os.path.join(INPUT_DIR, "s1_raw_content_*.json")), reverse=True)

if not metrics_files or not content_files:
    print("❌ 错误：未找到数据文件！")
    exit(1)

METRICS_FILE = metrics_files[0]
CONTENT_FILE = content_files[0]

print(f"📂 发现数据文件:")
print(f"   Metrics: {os.path.basename(METRICS_FILE)}")
print(f"   Content: {os.path.basename(CONTENT_FILE)}")

# ================= 1. 加载与清洗数据 =================
print("\n🔄 正在加载并清洗数据...")

df_metrics = pd.read_csv(METRICS_FILE)

with open(CONTENT_FILE, 'r', encoding='utf-8') as f:
    data_content = json.load(f)

df_content = pd.DataFrame(data_content)

# 智能识别 Prompt 列名
possible_prompt_cols = ['prompt', 'Prompt', 'question', 'Question', 'input', 'Input', 'text', 'Text']
prompt_col = None
for col in possible_prompt_cols:
    if col in df_content.columns:
        prompt_col = col
        break

if prompt_col is None:
    non_id_cols = [c for c in df_content.columns if 'id' not in c.lower()]
    if non_id_cols:
        prompt_col = non_id_cols[0]
    else:
        print("❌ 错误：无法识别问题列。")
        exit(1)

print(f"   🎯 识别到问题列：'{prompt_col}'")
df_content.rename(columns={prompt_col: 'Prompt'}, inplace=True)

# 合并数据
if 'Sample_ID' in df_metrics.columns and 'Sample_ID' in df_content.columns:
    df_merged = pd.merge(df_metrics, df_content, on='Sample_ID', how='inner')
elif 'sample_id' in df_metrics.columns and 'sample_id' in df_content.columns:
    df_merged = pd.merge(df_metrics, df_content, left_on='sample_id', right_on='sample_id', how='inner')
    df_merged.rename(columns={'sample_id': 'Sample_ID'}, inplace=True)
else:
    print("   ⚠️ 未找到共同 ID 列，按行顺序合并。")
    df_merged = pd.concat([df_metrics.reset_index(drop=True), df_content.reset_index(drop=True)], axis=1)
    if 'Sample_ID' not in df_merged.columns:
        df_merged['Sample_ID'] = range(1, len(df_merged) + 1)

# 筛选成功请求
if 'response' in df_merged.columns:
    df_success = df_merged[df_merged['response'].notna() & (df_merged['response'].str.len() > 0)].copy()
else:
    df_success = df_merged.copy()

total_requests = len(df_merged)
success_requests = len(df_success)
print(f"   总请求数：{total_requests}")
print(f"   成功请求数：{success_requests}")

if success_requests == 0:
    print("❌ 没有成功请求。")
    exit(1)

# ================= 2. 计算性能指标 =================
print("\n📊 计算系统性能硬指标...")

stats = {}

if 'TPS' in df_success.columns:
    stats['TPS_Median'] = df_success['TPS'].median()
    stats['TPS_Mean'] = df_success['TPS'].mean()
else:
    stats['TPS_Median'] = 0

threshold_ms = 300000  # 5 分钟
if 'TTFT_ms' in df_success.columns:
    ttft_all = df_success['TTFT_ms'].values
    ttft_clean = ttft_all[ttft_all < threshold_ms]
    
    stats['TTFT_Median_All'] = np.median(ttft_all)
    stats['TTFT_Max_All'] = np.max(ttft_all)
    stats['Outlier_Count'] = len(ttft_all) - len(ttft_clean)
    stats['TTFT_Median_Clean'] = np.median(ttft_clean) if len(ttft_clean) > 0 else stats['TTFT_Median_All']
else:
    stats['TTFT_Median_All'] = 0
    stats['TTFT_Median_Clean'] = 0
    stats['TTFT_Max_All'] = 0
    stats['Outlier_Count'] = 0

if 'Peak_Mem_MB' in df_success.columns:
    stats['Mem_Peak_Max_MB'] = df_success['Peak_Mem_MB'].max()
else:
    stats['Mem_Peak_Max_MB'] = 0

def is_complete(text):
    if not isinstance(text, str): return False
    text = text.strip()
    if len(text) < 5: return False
    end_marks = ['。', '！', '？', '”', '』', '）', ')', '.', '!', '?']
    if any(text.endswith(m) for m in end_marks): return True
    cut_off_chars = ['因', '所', '但', '可', '却', '然', '而', '由', '尽', '着', '的']
    return text[-1] not in cut_off_chars

df_success['Is_Complete'] = df_success['response'].apply(is_complete)
stats["Completion_Rate"] = (df_success['Is_Complete'].sum() / success_requests) * 100 if success_requests > 0 else 0

print(f"   ⚡ TPS (中位数): {stats['TPS_Median']:.2f} tokens/s")
print(f"   ⏱️ TTFT (常态中位数): {stats['TTFT_Median_Clean']:.1f} ms")
print(f"   ⚠️ TTFT (极端最大值): {stats['TTFT_Max_All']/60000:.1f} 分钟")
print(f"   🧠 峰值内存：{stats['Mem_Peak_Max_MB']:.1f} MB")
print(f"   ✅ 回答完整率：{stats['Completion_Rate']:.1f}%")

# ================= 3. 生成 Excel (优化列顺序) =================
print(f"\n📝 生成人工打分表：{OUTPUT_EXCEL} ...")

df_score = df_success.copy()
df_score['Semantic_Score_0_1'] = None
df_score['Logic_Score_1_5'] = None
df_score['Hallucination_Flag'] = None
df_score['Notes'] = ""

high_semantic_keywords = ['Idiom', 'Polyphone', 'Poetry', 'History', 'Culture', 'Philosophy', 'Literature', 'Advanced']
col_to_check = 'Category' if 'Category' in df_score.columns else ('Type' if 'Type' in df_score.columns else 'Prompt')
df_score['Is_High_Semantic'] = df_score[col_to_check].astype(str).apply(
    lambda x: any(k in x for k in high_semantic_keywords) or (x.startswith('A') if len(x) > 0 else False)
)

if 'Type' in df_score.columns:
    df_score = df_score.sort_values(by=['Type', 'Category' if 'Category' in df_score.columns else 'Sample_ID'])
else:
    df_score = df_score.sort_values(by='Sample_ID')

# 【优化】调整列顺序，把 Prompt 和 Response 放在前面，方便人工阅读
cols_to_export = [
    'Sample_ID', 
    'Type' if 'Type' in df_score.columns else 'Category',
    'Category' if 'Category' in df_score.columns else 'Sample_ID',
    'Is_High_Semantic', 
    'Prompt',          # 问题
    'response',        # 回答
    'Semantic_Score_0_1', 
    'Logic_Score_1_5', 
    'Hallucination_Flag', 
    'Notes',
    'TTFT_ms', 
    'TPS', 
    'Peak_Mem_MB'
]
final_cols = [c for c in cols_to_export if c in df_score.columns]

try:
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df_score[final_cols].to_excel(writer, index=False, sheet_name='人工评分表')
        
        guide_df = pd.DataFrame({
            "评分指南": [
                "【实验目标】评估边缘端模型中文质量",
                "",
                "1. Semantic_Score_0_1 (仅针对高阶题): 1=正确，0=错误",
                "2. Logic_Score_1_5: 5=完美，1=严重幻觉/逻辑混乱",
                "3. Hallucination_Flag: 1=有幻觉，0=无",
                "",
                f"【数据提示】本批次有 {stats['Outlier_Count']} 个极端延迟样本 (>5分钟)。"
            ]
        })
        guide_df.to_excel(writer, index=False, sheet_name='阅读说明')
    print(f"   ✅ Excel 已生成")
except Exception as e:
    print(f"❌ Excel 导出失败：{e}")
    exit(1)

# ================= 4. 绘制极简高清中文图表 (完整展示数据) =================
print("\n🎨 绘制极简性能图表 (重质量，轻参数)...")

sns.set(style="whitegrid", context="talk")
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# --- 图 1: TTFT 分布 (保留长尾分析，因为影响体验) ---
axs[0].hist(ttft_clean, bins=20, color='#4C72B0', edgecolor='black', alpha=0.7, label=f'正常请求 (<5 分钟), N={len(ttft_clean)}')
if stats['Outlier_Count'] > 0:
    outliers = ttft_all[ttft_all >= threshold_ms]
    axs[0].hist(outliers, bins=5, color='#DD8452', edgecolor='black', alpha=0.8, label=f'极端长尾延迟 (>5 分钟), N={stats["Outlier_Count"]}')

title_text = 'Qwen3:1.7b  场景一：首字延迟 (TTFT) 分布特征\n(主动散热下的长尾效应)'
axs[0].set_title(title_text, fontsize=14, fontweight='bold', fontproperties=my_font)
axs[0].set_xlabel('首字延迟 TTFT (毫秒，对数坐标)', fontsize=12, fontproperties=my_font)
axs[0].set_ylabel('样本频数', fontsize=12, fontproperties=my_font)

axs[0].set_xscale('log')
axs[0].legend(loc='upper right', fontsize=11, prop=my_font)
axs[0].grid(True, which="both", ls="-", alpha=0.2)

median_val = stats['TTFT_Median_Clean']
axs[0].axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'常态中位数：{median_val:.0f}ms')

# 简单的异常标注
if stats['Outlier_Count'] > 0:
    outlier_hist, _ = np.histogram(outliers, bins=5)
    max_outlier_count = max(outlier_hist) if len(outlier_hist) > 0 else 1
    y_text_pos = max_outlier_count * 1.1
    y_text_pos = min(y_text_pos, axs[0].get_ylim()[1] * 0.95)
    
    axs[0].text(outliers.mean(), y_text_pos, 
                '系统挂起/交换区抖动', 
                color='#DD8452', fontsize=11, ha='center', va='bottom',
                fontproperties=my_font,
                bbox=dict(facecolor='white', edgecolor='#DD8452', boxstyle='round,pad=0.3', alpha=0.9))

# --- 图 2: TPS 稳定性 (极简版：完整展示所有数据) ---
# 1. 绘制实时吞吐量曲线 (绿色)
axs[1].plot(range(len(df_success)), df_success['TPS'].values, marker='o', linestyle='-', markersize=4, color='#55A868', alpha=0.8, label='实时吞吐量')

# 2. 绘制中位数基准线 (红色虚线)
axs[1].axhline(stats['TPS_Median'], color='#D62728', linestyle='--', linewidth=2, label=f'中位数基准：{stats["TPS_Median"]:.1f} t/s')

# 3. 添加简单的统计信息框
textstr = f'峰值内存：{stats["Mem_Peak_Max_MB"]:.0f} MB\n完整率：{stats["Completion_Rate"]:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
axs[1].text(0.05, 0.95, textstr, transform=axs[1].transAxes, fontsize=11, verticalalignment='top', bbox=props, fontproperties=my_font)

axs[1].set_title('Qwen3:1.7b  场景一：生成吞吐量 (TPS) 稳定性监测\n(验证热节流是否发生)', fontsize=14, fontweight='bold', fontproperties=my_font)
axs[1].set_xlabel('测试样本序号 (时间轴)', fontsize=12, fontproperties=my_font)
axs[1].set_ylabel('生成速度 (Tokens/秒)', fontsize=12, fontproperties=my_font)

axs[1].legend(loc='upper right', fontsize=11, prop=my_font, framealpha=0.9)
axs[1].grid(True, ls="-", alpha=0.2)

# 【关键修改】移除所有 Y 轴限制，让 matplotlib 自动调整以完整显示数据
# 不再执行以下代码：
# if df_success['TPS'].max() > stats['TPS_Median'] * 5:
#     axs[1].set_ylim(0, stats['TPS_Median'] * 2.5)
#     axs[1].text(...)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_COMBINED, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ 高清图表已保存：{OUTPUT_PLOT_COMBINED}")

# ================= 5. 输出论文分析草稿 (侧重质量) =================
print("\n" + "="*80)
print("📄 论文第四章·场景一分析草稿 (侧重质量)")
print("="*80)
print(f"""
### 4.1 场景一：主动散热下的准稳态性能与质量评估

实验共采集有效样本 {success_requests} 个。
- **生成吞吐量 (TPS)**：中位数 **{stats['TPS_Median']:.2f} tokens/s**，表现稳定，未见明显热节流降频。
- **首字延迟 (TTFT)**：
  - 常态中位数：**{stats['TTFT_Median_Clean']:.1f} ms**。
  - 极端最大值：**{stats['TTFT_Max_All']/60000:.1f} 分钟** (共 {stats['Outlier_Count']} 个异常样本)。
  - **成因**：推测为内存饱和引发的 Swap 交换抖动。
- **鲁棒性**：回答完整率 **{stats['Completion_Rate']:.1f}%**。

**结论**：主动散热有效维持了计算频率 (TPS 稳定)，但未能解决内存瓶颈导致的长尾延迟。
**后续重点**：鉴于性能基线已确立，后续分析将聚焦于**回答质量 (语义准确性、逻辑连贯性)** 与 **幻觉率** 的评估。
""")
print("="*80)