import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# ================= 配置区域 =================
FILE_KEYWORDS = {
    "Qwen3-1.7B": "Qwen_s1_manual_evaluation",
    "Llama3.2-1B": "Llama_s1_manual_evaluation"
}
OUTPUT_CSV = "paper_combined_metrics.csv"
OUTPUT_DIR = "paper_charts"

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 数据加载与合并 =================
def load_and_merge_data():
    all_dfs = []
    print("📂 正在扫描并读取数据文件...")
    
    for model_name, keyword in FILE_KEYWORDS.items():
        matches = glob.glob(f"{keyword}*.xlsx") + glob.glob(f"{keyword}*.xls")
        if not matches:
            print(f"⚠️ 警告：未找到包含 '{keyword}' 的文件。")
            continue
            
        file_path = matches[0]
        print(f"   📄 发现文件: {file_path} -> 标记为: {model_name}")
        
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            df['Model'] = model_name
            
            col_mapping = {
                'Semantic_Score_0_1': 'semantic',
                'Logic_Score_1_5': 'logic',
                'Hallucination_Flag': 'hallucination',
                'TTFT_ms': 'ttft',
                'TPS': 'tps',
                'Peak_Mem_MB': 'memory',
                'response': 'response',
                'Sample_ID': 'sample_id'
            }
            
            rename_dict = {}
            for original, standard in col_mapping.items():
                if original in df.columns:
                    rename_dict[original] = standard
            
            df.rename(columns=rename_dict, inplace=True)
            all_dfs.append(df)
            
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
    
    if not all_dfs:
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ 数据合并完成。总样本数: {len(combined_df)}")
    
    numeric_cols = ['semantic', 'logic', 'hallucination', 'ttft', 'tps', 'memory']
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        else:
            combined_df[col] = np.nan
            
    return combined_df

# ================= 指标计算 =================
def calculate_metrics(df):
    if df is None or df.empty:
        return pd.DataFrame()
        
    metrics_list = []
    groups = df.groupby('Model')
    
    for name, group in groups:
        n = len(group)
        if n == 0: continue
        
        s_semantic = group['semantic'].mean() if 'semantic' in group else 0
        s_logic = group['logic'].mean() if 'logic' in group else 0
        d_hallu = group['hallucination'].mean() if 'hallucination' in group else 0
        
        r_complete = 1.0
        if 'response' in group:
            valid_count = group['response'].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 5).sum()
            r_complete = valid_count / n

        # --- 【关键修改】放宽 TTFT 过滤阈值到 60 秒 (60000ms) ---
        # 只过滤掉明显错误的极端值（比如 > 5 分钟），保留真实的高延迟
        ttft_clean = group[group['ttft'] < 300000]['ttft'] if 'ttft' in group else pd.Series()
        avg_ttft = ttft_clean.mean() if not ttft_clean.empty else 0
        
        # TPS 过滤保持不变
        tps_clean = group[(group['tps'] > 0) & (group['tps'] < 200)]['tps'] if 'tps' in group else pd.Series()
        avg_tps = tps_clean.mean() if not tps_clean.empty else 0
        
        avg_mem = group['memory'].mean() if 'memory' in group else 0
        
        perfect = flawed = error = 0
        if 'semantic' in group and 'hallucination' in group:
            perfect = group[(group['semantic'] == 1) & (group['hallucination'] == 0)].shape[0] / n
            flawed = group[(group['semantic'] == 1) & (group['hallucination'] == 1)].shape[0] / n
        if 'semantic' in group:
            error = group[group['semantic'] == 0].shape[0] / n
            
        metrics_list.append({
            'Model': name,
            'Count': n,
            'S_semantic': s_semantic,
            'S_logic': s_logic,
            'D_hallu': d_hallu,
            'R_complete': r_complete,
            'Avg_TTFT_ms': avg_ttft,
            'Avg_TPS': avg_tps,
            'Avg_Mem_MB': avg_mem,
            'Ratio_Perfect': perfect,
            'Ratio_Flawed': flawed,
            'Ratio_Error': error
        })
        
    return pd.DataFrame(metrics_list)

# ================= 绘图函数 =================

def plot_radar_6dim(metrics_df):
    """图1：六维能力雷达图"""
    if len(metrics_df) < 2: return
    print("🎨 绘制六维雷达图...")
    
    labels = np.array([
        '语义准确度\n(0-1)', 
        '逻辑适配度\n(1-5 归一化)', 
        '低幻觉\n(1-D_hallu)', 
        '回答完整率\n(0-1)',
        '推理速度\n(TPS 归一化)', 
        '低延迟\n(1/TTFT 归一化)'
    ])
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ['#E63946', '#457B9D']
    
    max_tps = metrics_df['Avg_TPS'].max() if metrics_df['Avg_TPS'].max() > 0 else 1
    min_tps = metrics_df['Avg_TPS'].min()
    max_ttft = metrics_df['Avg_TTFT_ms'].max() if metrics_df['Avg_TTFT_ms'].max() > 0 else 1
    min_ttft = metrics_df['Avg_TTFT_ms'].min()
    
    for i, row in metrics_df.iterrows():
        val_semantic = row['S_semantic']
        val_logic = row['S_logic'] / 5.0
        val_hallu = 1.0 - row['D_hallu']
        val_complete = row['R_complete']
        val_tps = (row['Avg_TPS'] - min_tps) / (max_tps - min_tps) if max_tps != min_tps else 0.5
        # 注意：如果 TTFT 差异巨大，归一化后低延迟项可能会压缩得很小，这是正常的
        val_ttft = (max_ttft - row['Avg_TTFT_ms']) / (max_ttft - min_ttft) if max_ttft != min_ttft else 0.5
        
        data = [val_semantic, val_logic, val_hallu, val_complete, val_tps, val_ttft]
        data += data[:1]
        
        color = colors[i % len(colors)]
        ax.plot(angles, data, color=color, linewidth=2.5, label=row['Model'])
        ax.fill(angles, data, color=color, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_title('场景一：模型综合能力六维雷达图', fontsize=16, pad=30, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)
    ax.set_yticklabels([])
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_radar_6dim.png'), dpi=300, bbox_inches='tight')
    print("   ✅ fig1_radar_6dim.png")
    plt.close()

def plot_quality_bars_grouped(metrics_df):
    """图2：分组柱状图"""
    print("🎨 绘制分组柱状图...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    rects1 = ax1.bar(x - width, metrics_df['S_semantic'], width, label='语义准确度 (0-1)', color='#2A9D8F')
    rects2 = ax1.bar(x, metrics_df['R_complete'], width, label='回答完整率', color='#E9C46A')
    
    ax1.set_ylabel('Score / Rate', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'], fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_title('场景一：语义理解与完整性对比', fontsize=14)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.02), ncol=2)
    
    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, metrics_df['S_logic'], width, label='逻辑适配度 (1-5)', color='#E76F51', alpha=0.8)
    ax2.set_ylabel('Logic Score (1-5)', color='#E76F51', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E76F51')
    ax2.set_ylim(0, 5.5)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.02))
    
    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#E76F51')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_quality_bars_grouped.png'), dpi=300, bbox_inches='tight')
    print("   ✅ fig2_quality_bars_grouped.png")
    plt.close()

def plot_error_stack(metrics_df):
    """图3：堆叠图"""
    print("🎨 绘制质量分布堆叠图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df))
    width = 0.5
    
    p1 = ax.bar(x, metrics_df['Ratio_Perfect'], width, label='完美回答 (无幻觉)', color='#2A9D8F')
    p2 = ax.bar(x, metrics_df['Ratio_Flawed'], width, bottom=metrics_df['Ratio_Perfect'], label='瑕不掩瑜 (核心对+幻觉)', color='#E9C46A')
    p3 = ax.bar(x, metrics_df['Ratio_Error'], width, bottom=metrics_df['Ratio_Perfect']+metrics_df['Ratio_Flawed'], label='严重错误 (核心错)', color='#E76F51')
    
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('场景一：回答质量分布分析', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    for i, row in metrics_df.iterrows():
        if row['Ratio_Perfect'] > 0.05:
            ax.text(i, row['Ratio_Perfect']/2, f"{row['Ratio_Perfect']:.1%}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        if row['Ratio_Flawed'] > 0.05:
            ax.text(i, row['Ratio_Perfect'] + row['Ratio_Flawed']/2, f"{row['Ratio_Flawed']:.1%}", ha='center', va='center', color='black', fontsize=10)
        if row['Ratio_Error'] > 0.05:
            ax.text(i, row['Ratio_Perfect'] + row['Ratio_Flawed'] + row['Ratio_Error']/2, f"{row['Ratio_Error']:.1%}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_error_distribution.png'), dpi=300, bbox_inches='tight')
    print("   ✅ fig3_error_distribution.png")
    plt.close()

def plot_performance_bars_realistic(metrics_df):
    """图4：真实性能图（保留高延迟 + 修复图例遮挡）"""
    print("🎨 绘制性能指标图（修复图例布局）...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, metrics_df['Avg_TPS'], width, label='生成吞吐量 (TPS)', color='#457B9D')
    ax1.set_ylabel('Tokens Per Second', color='#457B9D', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#457B9D')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'], fontsize=12)
    ax1.set_title('场景一：系统性能硬指标对比', fontsize=14)
    
    # 👇 关键修复：将图例移到左下角，避免遮挡
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=True, fancybox=True, shadow=False)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, metrics_df['Avg_TTFT_ms'], width, label='首字延迟 (TTFT/ms)', color='#E76F51', alpha=0.8)
    ax2.set_ylabel('Time To First Token (ms)', color='#E76F51', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E76F51')
    
    # 👇 TTFT 图例也移到右下角，与 TPS 图例对称
    ax2.legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True, fancybox=True, shadow=False)
    
    # 添加数值标签
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        if h > 1000:
            label = f'{h/1000:.1f}s'
        else:
            label = f'{h:.1f}'
        ax2.annotate(label, xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#E76F51')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_performance_fixed_layout.png'), dpi=300, bbox_inches='tight')
    print("   ✅ fig4_performance_fixed_layout.png")
    plt.close()

# ================= 主程序 =================
def main():
    print("🚀 启动最终版分析流程（保留高延迟）...")
    df = load_and_merge_data()
    if df is None: return
    
    metrics_df = calculate_metrics(df)
    if metrics_df.empty:
        print("❌ 计算结果为空。")
        return
        
    print("\n📊 === 核心指标统计摘要 ===")
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    display_cols = ['Model', 'Count', 'S_semantic', 'S_logic', 'D_hallu', 'Avg_TPS', 'Avg_TTFT_ms']
    available_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[available_cols].to_string(index=False))
    
    if metrics_df['S_semantic'].isna().any():
        print("\n⚠️ 警告：检测到 NaN 值。")
    else:
        print("\n✅ 数据校验通过。")

    metrics_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细数据表已导出: {OUTPUT_CSV}")
    
    plot_radar_6dim(metrics_df)
    plot_quality_bars_grouped(metrics_df)
    plot_error_stack(metrics_df)
    plot_performance_bars_realistic(metrics_df)  # 使用新版本
    
    print(f"\n🎉 所有任务完成！图表已保存至 './{OUTPUT_DIR}/' 目录。")
    print("📝 提示：请在论文中解释 Qwen 模型 TTFT 较高的原因（如模型加载机制、显存瓶颈等）。")

if __name__ == "__main__":
    main()