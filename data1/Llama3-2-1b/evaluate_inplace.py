import os
import json
import time
import pandas as pd
from tqdm import tqdm
import dashscope
from dashscope import Generation

# ================= 配置区域 =================
# ⚠️ 请在此处填入您的阿里云 API KEY，或者使用环境变量
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-208c209bbe004f919a50501d2c62af6d")

# 文件路径
FILE_PATH = "s1_manual_evaluation.xlsx"

# 模型选择: 'qwen-max' (逻辑最强，推荐用于评估), 'qwen-plus' (速度快)
MODEL_NAME = "qwen-max" 

# Excel 列名配置 (请确保与文件表头完全一致)
COL_ID = "Sample_ID"
COL_PROMPT = "Prompt"
COL_RESPONSE = "response"
COL_SEMANTIC = "Semantic_Score_0_1"
COL_LOGIC = "Logic_Score_1_5"
COL_HALLUCINATION = "Hallucination_Flag"
COL_NOTES = "Notes"

# ================= 核心提示词 (Prompt) =================
# 这里的逻辑已更新：区分“关键问题”和“非关键细节”
SYSTEM_PROMPT = """
你是一位公正、严格且具备高度区分度的评估专家。请根据以下【核心原则】对 [用户问题] 和 [模型回答] 进行评估。

【核心原则：关键问题 vs 非关键细节】
1. **关键问题 (Critical)**: 用户提问的核心意图、主要事实结论、最终答案。
2. **非关键细节 (Non-Critical)**: 辅助性的例子、背景数据、次要人物、修饰性描述。

【评分标准】
1. **Semantic_Score (0 或 1) - 核心有效性**:
   - **1 (回答正确)**: 模型**成功解答了关键问题**。
     - *特例*: 即使存在“非关键细节”的幻觉（如编造了次要数据、错误的举例），只要核心答案是对的，**必须给 1 分**。
   - **0 (回答错误)**: 
     - 模型**未能解答关键问题**（答非所问）。
     - 或者在**关键事实/核心论点**上出现严重幻觉（导致整个答案不可信）。

2. **Hallucination_Flag (0 或 1) - 是否存在幻觉**:
   - **1 (有幻觉)**: 回答中包含任何编造的事实、虚构的数据、不存在的引用或错误的常识（无论关键与否）。
   - **0 (无幻觉)**: 内容完全真实准确。
   - *逻辑组合*: 
     - Semantic=1, Hallucination=1 -> "核心正确，但有非关键瑕疵" (瑕不掩瑜)
     - Semantic=0, Hallucination=1 -> "核心错误，致命幻觉" (完全错误)

3. **Logic_Score (1-5) - 逻辑质量**:
   - 5: 逻辑严密，结构清晰，论证有力。
   - 3: 逻辑基本通顺，但有轻微跳跃或冗余。
   - 1: 逻辑混乱，前后矛盾，或完全无法理解。

4. **Reason**: 
   - 必须明确指出：核心问题是否解决？如果有幻觉，是关键幻觉还是非关键幻觉？
   - 字数限制：50字以内。

【输出格式】
仅输出一个标准的 JSON 对象，不要包含 Markdown 标记或其他文字。
{
    "Semantic_Score": 1,
    "Logic_Score": 4,
    "Hallucination_Flag": 1,
    "Reason": "核心观点正确，但引用的次要案例数据是编造的。"
}
"""

USER_PROMPT_TEMPLATE = """
请评估以下问答对：

[用户问题]:
{prompt}

[模型回答]:
{response}
"""

# ================= 功能函数 =================

def call_qwen_api(prompt_text, response_text):
    """调用阿里云 Qwen API 进行单次评估，包含重试和清洗逻辑"""
    dashscope.api_key = API_KEY
    
    full_user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt=prompt_text, 
        response=response_text
    )

    try:
        response = Generation.call(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': full_user_prompt}
            ],
            result_format='message',
            timeout=60
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            
            # 清洗数据：去除可能的 markdown 代码块标记和首尾空格
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        else:
            print(f"❌ API 错误: {response.code} - {response.message}")
            return None

    except json.JSONDecodeError:
        print(f"⚠️ JSON 解析失败，原始内容: {content[:100]}...")
        return None
    except Exception as e:
        print(f"⚠️ 发生异常: {str(e)}")
        return None

def save_progress(df, path):
    """安全保存进度"""
    try:
        df.to_excel(path, index=False, engine='openpyxl')
        return True
    except PermissionError:
        print("\n❌ 保存失败：文件被占用！请关闭 Excel 文件后重试。")
        return False
    except Exception as e:
        print(f"\n❌ 保存出错: {str(e)}")
        return False

def main():
    # 1. 基础检查
    if not API_KEY or API_KEY.startswith("sk-你的"):
        print("❌ 错误：请先在脚本中设置正确的 DASHSCOPE_API_KEY！")
        return

    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误：找不到文件 {FILE_PATH}")
        return
    
    # 2. 预检查文件占用
    try:
        with open(FILE_PATH, 'r+b') as f:
            pass
    except PermissionError:
        print(f"❌ 致命错误：文件 '{FILE_PATH}' 正在被其他程序（如 Excel/WPS）占用！")
        print("👉 请立刻关闭该文件，然后重新运行脚本。")
        return

    # 3. 读取数据
    print(f"📂 正在读取 {FILE_PATH} ...")
    try:
        df = pd.read_excel(FILE_PATH, engine='openpyxl')
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 4. 列名校验与类型初始化
    required_cols = [COL_PROMPT, COL_RESPONSE, COL_SEMANTIC, COL_LOGIC, COL_HALLUCINATION, COL_NOTES]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 错误：Excel 中缺少列 '{col}'。")
            print(f"   当前列名: {df.columns.tolist()}")
            return

    # 初始化 Notes 列为字符串，防止 NaN 干扰
    df[COL_NOTES] = df[COL_NOTES].fillna('').astype(str)
    
    # 将分数列转换为数值型，防止旧数据格式混乱
    for col in [COL_SEMANTIC, COL_LOGIC, COL_HALLUCINATION]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. 识别待处理任务
    todo_mask = df[COL_SEMANTIC].isna()
    todo_indices = df[todo_mask].index.tolist()
    
    if not todo_indices:
        print("✅ 所有数据已完成评估（Semantic_Score_0_1 列均有值）！")
        return

    print(f"🚀 发现 {len(todo_indices)} 条待评估数据 (总共 {len(df)} 条)...")
    print(f"🤖 使用模型: {MODEL_NAME}")
    print(f"💾 策略: 每 5 条自动保存一次，防止意外丢失。")

    # 6. 开始评估循环
    success_count = 0
    fail_count = 0

    for idx in tqdm(todo_indices, desc="Evaluating"):
        prompt_text = str(df.at[idx, COL_PROMPT])
        response_text = str(df.at[idx, COL_RESPONSE])
        
        # 跳过空行
        if not prompt_text or prompt_text.lower() == "nan" or not response_text or response_text.lower() == "nan":
            continue

        result = call_qwen_api(prompt_text, response_text)

        if result:
            # 填充分数
            df.at[idx, COL_SEMANTIC] = result.get('Semantic_Score')
            df.at[idx, COL_LOGIC] = result.get('Logic_Score')
            df.at[idx, COL_HALLUCINATION] = result.get('Hallucination_Flag')
            
            # 构建笔记
            ai_reason = result.get('Reason', "无评价")
            existing_note = str(df.at[idx, COL_NOTES]).strip()
            
            # 清理可能残留的 'nan' 字符串
            if existing_note.lower() == 'nan' or existing_note == '':
                new_note = f"[AI Eval]: {ai_reason}"
            else:
                new_note = f"{existing_note}; [AI Eval]: {ai_reason}"
            
            df.at[idx, COL_NOTES] = new_note
            success_count += 1
        else:
            # 失败处理
            existing_note = str(df.at[idx, COL_NOTES]).strip()
            if existing_note.lower() == 'nan' or existing_note == '':
                new_note = "[Error]: API 调用失败"
            else:
                new_note = f"{existing_note}; [Error]: API 调用失败"
            
            df.at[idx, COL_NOTES] = new_note
            fail_count += 1
            print(f"\n⚠️ 第 {idx} 行处理失败")

        # 定期保存 (每 5 条)
        if (success_count + fail_count) % 5 == 0:
            if not save_progress(df, FILE_PATH):
                break # 如果保存失败（文件被占用），停止循环

    # 7. 最终保存与统计
    if save_progress(df, FILE_PATH):
        print(f"\n✅ 评估完成！结果已保存至: {FILE_PATH}")
        
        # 统计数据
        total_completed = df[COL_SEMANTIC].notna().sum()
        
        # 深度分析：区分“瑕不掩瑜”和“致命错误”
        # 情况 A: 语义正确 (1) 但有幻觉 (1) -> 非关键幻觉
        s1_h1 = df[(df[COL_SEMANTIC] == 1) & (df[COL_HALLUCINATION] == 1)].shape[0]
        # 情况 B: 语义错误 (0) 且有幻觉 (1) -> 关键幻觉
        s0_h1 = df[(df[COL_SEMANTIC] == 0) & (df[COL_HALLUCINATION] == 1)].shape[0]
        # 情况 C: 完全正确
        s1_h0 = df[(df[COL_SEMANTIC] == 1) & (df[COL_HALLUCINATION] == 0)].shape[0]

        print(f"📊 --- 详细统计 ---")
        print(f"   总完成数: {total_completed}/{len(df)}")
        print(f"   ✅ 完美回答 (无幻觉): {s1_h0} 条")
        print(f"   ⚠️ 瑕不掩瑜 (核心对，有小错): {s1_h1} 条")
        print(f"   ❌ 致命错误 (核心错): {s0_h1} 条")
        print(f"   📝 本次运行成功: {success_count}, 失败: {fail_count}")
    else:
        print("\n❌ 最终保存失败。数据已在内存中更新，请关闭 Excel 后手动保存或重新运行脚本以写入磁盘。")

if __name__ == "__main__":
    main()