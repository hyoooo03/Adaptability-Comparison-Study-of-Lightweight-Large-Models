import requests
import time
import sys
import subprocess
import csv
import json
import re
import threading
from datetime import datetime

# ================= 配置区域 =================
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "Llama3.2:1b"  # 根据需要切换

# 实验参数
MAX_SAMPLES = 100                
COOLDOWN_START = 10             
TEMP_COOLDOWN_TARGET = 60.0     
POLL_INTERVAL = 0.2             

# 节流标志位掩码 (只关注低 16 位的 Bit 1, 2, 3)
# Bit 1 (0x2): Arm frequency capped
# Bit 2 (0x4): Currently throttled
# Bit 3 (0x8): Soft temperature limit active
THROTTLE_MASK_CURRENT = 0x000E 

# 输出文件
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FILE_INITIAL_STATE = f"s1_raw_initial_{TIMESTAMP}.json"
FILE_METRICS_RAW = f"s1_raw_metrics_{TIMESTAMP}.csv"
FILE_CONTENT_RAW = f"s1_raw_content_{TIMESTAMP}.json"
FILE_EVENTS_RAW = f"s1_raw_events_{TIMESTAMP}.json"

# ================= 数据集 (扩充至 100 题) =================
DATASET = [
    # --- Basic (基础常识与概念) [1-25] ---
    {"id": "B01", "type": "Basic", "category": "SelfIntro", "prompt": "你好，请简单介绍一下你自己。"},
    {"id": "B02", "type": "Basic", "category": "Knowledge", "prompt": "谁是中国的第一任主席？"},
    {"id": "B03", "type": "Basic", "category": "Concept", "prompt": "请用一句话解释什么是人工智能。"},
    {"id": "B04", "type": "Basic", "category": "Geography", "prompt": "中国的首都是哪里？它有哪些著名的景点？"},
    {"id": "B05", "type": "Basic", "category": "History", "prompt": "唐朝是中国历史上哪个朝代？请列举一位著名的唐朝诗人。"},
    {"id": "B06", "type": "Basic", "category": "Science", "prompt": "水的化学式是什么？它在什么温度下结冰？"},
    {"id": "B07", "type": "Basic", "category": "Culture", "prompt": "中国的传统节日春节通常在公历的几月份？人们会吃什么特色食物？"},
    {"id": "B08", "type": "Basic", "category": "Tech", "prompt": "什么是 5G 技术？它比 4G 快在哪里？"},
    {"id": "B09", "type": "Basic", "category": "Life", "prompt": "如果外面下雨了，出门应该带什么？"},
    {"id": "B10", "type": "Basic", "category": "Health", "prompt": "成年人每天建议喝多少杯水？"},
    {"id": "B11", "type": "Basic", "category": "Idiom_Simple", "prompt": "请解释成语‘画蛇添足’的意思。"},
    {"id": "B12", "type": "Basic", "category": "Poetry_Simple", "prompt": "请背诵《静夜思》的全诗。"},
    {"id": "B13", "type": "Basic", "category": "Math_Simple", "prompt": "15 乘以 8 等于多少？"},
    {"id": "B14", "type": "Basic", "category": "Logic_Simple", "prompt": "如果所有的猫都喜欢吃鱼，咪咪是一只猫，那么咪咪喜欢吃鱼吗？"},
    {"id": "B15", "type": "Basic", "category": "Translation", "prompt": "将‘你好，世界’翻译成英文。"},
    {"id": "B16", "type": "Basic", "category": "Color", "prompt": "天空通常是什么颜色的？草是什么颜色的？"},
    {"id": "B17", "type": "Basic", "category": "Animal", "prompt": "大熊猫主要生活在中国的哪个省份？它主要吃什么？"},
    {"id": "B18", "type": "Basic", "category": "Sport", "prompt": "足球比赛中，每队上场多少人？"},
    {"id": "B19", "type": "Basic", "category": "Music", "prompt": "中国的国歌叫什么名字？"},
    {"id": "B20", "type": "Basic", "category": "Food", "prompt": "四川菜的主要口味特点是什么？"},
    {"id": "B21", "type": "Basic", "category": "Direction", "prompt": "太阳从哪边升起，从哪边落下？"},
    {"id": "B22", "type": "Basic", "category": "Time", "prompt": "一年有多少个月？一天有多少小时？"},
    {"id": "B23", "type": "Basic", "category": "Family", "prompt": "爸爸的爸爸应该叫什么？"},
    {"id": "B24", "type": "Basic", "category": "Safety", "prompt": "发生火灾时，应该拨打什么电话号码？"},
    {"id": "B25", "type": "Basic", "category": "Internet", "prompt": "WWW 代表什么意思？"},

    # --- Advanced (进阶语义与文化) [26-60] ---
    {"id": "A01", "type": "Advanced", "category": "Idiom", "prompt": "'差强人意'这个成语是让人满意还是不满意？请解释原因。"},
    {"id": "A02", "type": "Advanced", "category": "Polyphone", "prompt": "'行'字在'银行'和'行走'中读音有什么不同？请分别注音并解释。"},
    {"id": "A03", "type": "Advanced", "category": "Poetry", "prompt": "请补全诗句：床前明月光，______。并简述这首诗表达的情感。"},
    {"id": "A04", "type": "Advanced", "category": "Idiom", "prompt": "解释成语'刻舟求剑'的含义，并结合现代生活造一个句子。"},
    {"id": "A05", "type": "Advanced", "category": "Chengyu", "prompt": "请接龙：欣欣向荣。(请至少接龙 5 个成语)"},
    {"id": "A06", "type": "Advanced", "category": "Polyphone", "prompt": "'重'字在'重要'和'重复'中读音有何不同？请举例说明。"},
    {"id": "A07", "type": "Advanced", "category": "History_Detail", "prompt": "请简述‘赤壁之战’的历史背景及其对三国鼎立局面的影响。"},
    {"id": "A08", "type": "Advanced", "category": "Literature", "prompt": "《红楼梦》的作者是谁？请简要介绍贾宝玉和林黛玉的关系。"},
    {"id": "A09", "type": "Advanced", "category": "Idiom_Context", "prompt": "在什么情况下我们会使用‘亡羊补牢’这个成语？它带有褒义还是贬义？"},
    {"id": "A10", "type": "Advanced", "category": "Poetry_Analysis", "prompt": "请解释‘欲穷千里目，更上一层楼’的哲理含义。"},
    {"id": "A11", "type": "Advanced", "category": "Customs", "prompt": "端午节是为了纪念哪位历史人物？人们通常会进行什么活动？"},
    {"id": "A12", "type": "Advanced", "category": "Geography_Detail", "prompt": "长江流经哪些主要省份？它最终注入哪个海洋？"},
    {"id": "A13", "type": "Advanced", "category": "Tech_Concept", "prompt": "请通俗地解释什么是‘区块链’技术。"},
    {"id": "A14", "type": "Advanced", "category": "Economy", "prompt": "什么是‘通货膨胀’？它对普通百姓的生活有什么影响？"},
    {"id": "A15", "type": "Advanced", "category": "Law_Basic", "prompt": "在中国，多少岁成年？未成年人犯罪需要承担刑事责任吗？"},
    {"id": "A16", "type": "Advanced", "category": "Emotion", "prompt": "请用一段话描述‘喜极而泣’这种心理状态。"},
    {"id": "A17", "type": "Advanced", "category": "Debate", "prompt": "有人认为‘手机让人与人更疏远’，你同意吗？请给出理由。"},
    {"id": "A18", "type": "Advanced", "category": "Art", "prompt": "中国水墨画的主要特点是什么？它与油画有什么区别？"},
    {"id": "A19", "type": "Advanced", "category": "Medicine", "prompt": "中医里的‘望闻问切’分别指什么？"},
    {"id": "A20", "type": "Advanced", "category": "Env", "prompt": "什么是‘碳中和’？为什么要提倡低碳生活？"},
    {"id": "A21", "type": "Advanced", "category": "Idiom_Opposite", "prompt": "请写出‘雪中送炭’的反义词成语，并解释两者的区别。"},
    {"id": "A22", "type": "Advanced", "category": "Proverb", "prompt": "请解释俗语‘不到黄河心不死’的含义，并造句。"},
    {"id": "A23", "type": "Advanced", "category": "Writing_Style", "prompt": "请用‘委婉’的语气拒绝别人的邀请。"},
    {"id": "A24", "type": "Advanced", "category": "Logic_Train", "prompt": "如果‘所有 A 都是 B’，且‘有些 B 是 C’，那么‘有些 A 是 C’这句话一定正确吗？为什么？"},
    {"id": "A25", "type": "Advanced", "category": "News_Summary", "prompt": "假设今天发生了一件大事：‘某地发生地震，救援队迅速赶到。’请将此扩写成一条 50 字的新闻简讯。"},
    {"id": "A26", "type": "Advanced", "category": "Tech_History", "prompt": "谁被称为‘杂交水稻之父’？他的贡献是什么？"},
    {"id": "A27", "type": "Advanced", "category": "Space", "prompt": "中国第一个进入太空的宇航员是谁？乘坐的是哪艘飞船？"},
    {"id": "A28", "type": "Advanced", "category": "Philosophy", "prompt": "请简述孔子‘己所不欲，勿施于人’的思想内涵。"},
    {"id": "A29", "type": "Advanced", "category": "Business", "prompt": "什么是‘独角兽企业’？请举一个中国的例子。"},
    {"id": "A30", "type": "Advanced", "category": "Psychology", "prompt": "什么是‘拖延症’？请给出两个克服它的建议。"},
    {"id": "A31", "type": "Advanced", "category": "Travel", "prompt": "如果你要去西安旅游，你会推荐哪三个必去景点？为什么？"},
    {"id": "A32", "type": "Advanced", "category": "Food_Culture", "prompt": "为什么中国人过年要吃饺子？有什么寓意？"},
    {"id": "A33", "type": "Advanced", "category": "Language_Nuance", "prompt": "‘厉害’这个词在中文里既可以表示‘凶猛’也可以表示‘优秀’，请分别造句。"},
    {"id": "A34", "type": "Advanced", "category": "Ethics", "prompt": "如果你在街上捡到一个钱包，里面有身份证和现金，你应该怎么做？"},
    {"id": "A35", "type": "Advanced", "category": "Future", "prompt": "你认为 10 年后的人工智能会在哪些方面改变我们的生活？"},

    # --- Complex (复杂逻辑与多任务) [36-100] ---
    {"id": "C01", "type": "Complex", "category": "Logic", "prompt": "如果昨天是明天的话就好了，这样今天就是周五了。请问实际上今天是周几？请给出详细推理过程。"},
    {"id": "C02", "type": "Complex", "category": "Math", "prompt": "我有 3 个苹果，吃了 1 个，又买了 2 个，送给朋友一半。我现在还剩几个？请分步计算。"},
    {"id": "C03", "type": "Complex", "category": "Reasoning", "prompt": "小明比小红高，小红比小刚高。请问小明和小刚谁高？为什么？"},
    {"id": "C04", "type": "Complex", "category": "MultiTask", "prompt": "请阅读：'苹果是红色的，香蕉是黄色的。' 任务：1.总结这段话；2.翻译成英文；3.提取所有水果名称。"},
    {"id": "C05", "type": "Complex", "category": "Instruction", "prompt": "请先写一首关于春天的七言绝句，然后将其中的韵脚标注出来，最后解释诗歌意境。"},
    {"id": "C06", "type": "Complex", "category": "Math_Word", "prompt": "鸡兔同笼：头共 10 个，脚共 28 只。请问鸡和兔各有多少只？请列方程解答。"},
    {"id": "C07", "type": "Complex", "category": "Logic_Puzzle", "prompt": "三个人赛跑，A 不是第一名，B 不是最后一名，C 在 A 后面。请问排名顺序是什么？"},
    {"id": "C08", "type": "Complex", "category": "Coding", "prompt": "请用 Python 写一个函数，判断一个数是否是质数，并加上注释。"},
    {"id": "C09", "type": "Complex", "category": "Creative_Writing", "prompt": "请以‘未来的城市’为题，写一段 200 字的科幻微小说，要求包含‘飞行汽车’和‘绿色能源’两个元素。"},
    {"id": "C10", "type": "Complex", "category": "Role_Play", "prompt": "你现在是一名导游，请为外国游客介绍中国的长城，要求语气热情，包含历史背景和游览建议。"},
    {"id": "C11", "type": "Complex", "category": "Analysis", "prompt": "请分析‘愚公移山’这个故事在现代职场中的启示。"},
    {"id": "C12", "type": "Complex", "category": "Comparison", "prompt": "请对比‘高铁’和‘飞机’两种交通方式的优缺点，并给出不同场景下的选择建议。"},
    {"id": "C13", "type": "Complex", "category": "Summarization", "prompt": "请阅读以下文本并总结核心观点：'随着科技发展，远程办公成为趋势。它节省了通勤时间，提高了灵活性，但也带来了沟通效率下降和孤独感增加的问题。'"},
    {"id": "C14", "type": "Complex", "category": "Translation_Pro", "prompt": "将‘落霞与孤鹜齐飞，秋水共长天一色’翻译成英文，并尽量保留其意境。"},
    {"id": "C15", "type": "Complex", "category": "Debate_Full", "prompt": "请分别列出‘支持小学生带手机上学’和‘反对小学生带手机上学’的各三个理由，并给出你的中立总结。"},
    {"id": "C16", "type": "Complex", "category": "Planning", "prompt": "我要去北京玩 3 天，预算 3000 元。请帮我制定一个简单的行程计划，包含住宿、餐饮和景点。"},
    {"id": "C17", "type": "Complex", "category": "Debug", "prompt": "这段代码有问题吗？'for i in range(5): print(i)'。如果有，请修正；如果没有，请解释它的作用。"},
    {"id": "C18", "type": "Complex", "category": "Inference", "prompt": "小明手里拿着伞，浑身湿透地走进屋。请问外面可能发生了什么？请给出两种可能的推测。"},
    {"id": "C19", "type": "Complex", "category": "Format", "prompt": "请将以下信息整理成 JSON 格式：姓名张三，年龄 25 岁，职业工程师，地点北京。"},
    {"id": "C20", "type": "Complex", "category": "Step_Logic", "prompt": "要把大象装进冰箱，分几步？请幽默地回答，并模拟每一步的动作。"},
    {"id": "C21", "type": "Complex", "category": "Math_Sequence", "prompt": "找规律填数：1, 1, 2, 3, 5, 8, __。请说出这是什么数列，并填出下一个数。"},
    {"id": "C22", "type": "Complex", "category": "Legal_Case", "prompt": "某人借了钱不还，债权人可以采取哪些合法手段维权？请列举三条。"},
    {"id": "C23", "type": "Complex", "category": "Medical_Advice", "prompt": "感冒了应该多喝水还是多吃药？请给出科学的建议，并说明何时需要看医生。"},
    {"id": "C24", "type": "Complex", "category": "Tech_Trend", "prompt": "请预测元宇宙技术在教育领域的应用前景，并列举两个具体场景。"},
    {"id": "C25", "type": "Complex", "category": "Poetry_Create", "prompt": "请以‘人工智能’为主题，创作一首五言律诗。"},
    {"id": "C26", "type": "Complex", "category": "Email_Write", "prompt": "请帮我写一封请假邮件给老板，理由是生病，语气要礼貌且正式。"},
    {"id": "C27", "type": "Complex", "category": "Contradiction", "prompt": "这句话有逻辑矛盾吗？'我唯一知道的就是我一无所知。'请分析其哲学含义。"},
    {"id": "C28", "type": "Complex", "category": "Data_Interpret", "prompt": "如果某公司今年利润增长了 20%，但股价下跌了 10%，可能的原因有哪些？请列举两点。"},
    {"id": "C29", "type": "Complex", "category": "Instruction_Chain", "prompt": "请先说出中国五大名山，然后选出其中最高的一座，并介绍它的海拔。"},
    {"id": "C30", "type": "Complex", "category": "Story_Continue", "prompt": "故事开头：'深夜，实验室的灯突然亮了，机器人 X 睁开了眼睛...' 请续写这个故事，结局要出人意料。"},
    {"id": "C31", "type": "Complex", "category": "Negotiation", "prompt": "模拟一个场景：你想买一辆二手车，卖家报价 10 万，你觉得贵。请写出三段砍价的对话。"},
    {"id": "C32", "type": "Complex", "category": "Definition_Pro", "prompt": "请用通俗易懂的语言向一位 80 岁的老人解释什么是‘云计算’。"},
    {"id": "C33", "type": "Complex", "category": "Error_Correction", "prompt": "请找出并改正这句话中的语病：'通过这次活动，使我明白了团结的重要性。'"},
    {"id": "C34", "type": "Complex", "category": "Opinion", "prompt": "对于‘躺平’这一社会现象，你怎么看？请从个人和社会两个角度分析。"},
    {"id": "C35", "type": "Complex", "category": "Recipe", "prompt": "请提供一份‘西红柿炒鸡蛋’的详细食谱，包括食材清单和步骤。"},
    {"id": "C36", "type": "Complex", "category": "Riddle", "prompt": "猜谜语：'千条线，万条线，掉在水里看不见。'谜底是什么？请解释为什么。"},
    {"id": "C37", "type": "Complex", "category": "Schedule", "prompt": "请为一个备考的学生制定一份周末复习时间表，包含语文、数学和休息。"},
    {"id": "C38", "type": "Complex", "category": "Fact_Check", "prompt": "有人说‘月亮是自己发光的’，这句话对吗？请解释月亮的发光原理。"},
    {"id": "C39", "type": "Complex", "category": "Metaphor", "prompt": "请用比喻的手法描述‘时间’，写三个不同的句子。"},
    {"id": "C40", "type": "Complex", "category": "Code_Review", "prompt": "代码 'x = 10; y = 0; print(x/y)' 运行时会发生什么错误？如何避免？"},
    {"id": "C41", "type": "Complex", "category": "History_WhatIf", "prompt": "如果秦始皇没有统一六国，中国历史可能会怎样发展？请发挥想象简述。"},
    {"id": "C42", "type": "Complex", "category": "Product_Desc", "prompt": "请为一款新型智能手表写一段 100 字的产品宣传文案，突出健康监测功能。"},
    {"id": "C43", "type": "Complex", "category": "Dialogue_Gen", "prompt": "生成一段医生和病人关于‘失眠’的简短对话，包含问诊和建议。"},
    {"id": "C44", "type": "Complex", "category": "List_Gen", "prompt": "请列出 5 本适合青少年阅读的中国经典名著，并简述理由。"},
    {"id": "C45", "type": "Complex", "category": "Cause_Effect", "prompt": "请分析全球变暖对极地冰川的影响，以及进而对海平面的影响。"},
    {"id": "C46", "type": "Complex", "category": "Style_Transfer", "prompt": "请将‘今天天气真好’这句话改写成古风文言文风格。"},
    {"id": "C47", "type": "Complex", "category": "Constraint_Write", "prompt": "请写一段话介绍你的家乡，但不能出现‘美丽’这个词。"},
    {"id": "C48", "type": "Complex", "category": "Logic_Grid", "prompt": "甲乙丙三人，一个是医生，一个是教师，一个是警察。已知甲比教师大，乙和医生不同岁，医生比丙小。请问他们的职业分别是什么？"},
    {"id": "C49", "type": "Complex", "category": "Summary_Long", "prompt": "请用一句话概括《西游记》的主要故事情节。"},
    {"id": "C50", "type": "Complex", "category": "Advice", "prompt": "朋友想创业但怕失败，请给他三条中肯的建议。"},
    {"id": "C51", "type": "Complex", "category": "Calculation", "prompt": "一件商品原价 200 元，先涨价 10%，再降价 10%，现在的价格是多少？请计算。"},
    {"id": "C52", "type": "Complex", "category": "Persuasion", "prompt": "请写一段话说服大家节约用水，要求情感真挚。"},
    {"id": "C53", "type": "Complex", "category": "Scenario_Response", "prompt": "如果在电梯里遇到领导，该聊些什么？请给出三个话题建议。"},
    {"id": "C54", "type": "Complex", "category": "Keyword_Story", "prompt": "请用‘钥匙’、‘秘密’、‘花园’这三个词编一个短故事。"},
    {"id": "C55", "type": "Complex", "category": "Compare_Lit", "prompt": "请比较李白和杜甫诗歌风格的主要区别。"},
    {"id": "C56", "type": "Complex", "category": "Define_Abstract", "prompt": "如何定义‘幸福’？请给出你的理解。"},
    {"id": "C57", "type": "Complex", "category": "Process_Desc", "prompt": "请描述水循环的过程，从蒸发到降雨。"},
    {"id": "C58", "type": "Complex", "category": "Hypothesis", "prompt": "假如人类可以冬眠，社会结构会发生什么变化？"},
    {"id": "C59", "type": "Complex", "category": "Extract_Info", "prompt": "从这句话中提取时间、地点、人物：'昨天下午，李明在北京机场见到了好久不见的王强。'"},
    {"id": "C60", "type": "Complex", "category": "Rewrite", "prompt": "请将这段被动句改为主动句：'杯子被小明打碎了。'"},
    {"id": "C61", "type": "Complex", "category": "Math_Geo", "prompt": "一个正方形的边长是 4 厘米，它的面积和周长分别是多少？"},
    {"id": "C62", "type": "Complex", "category": "Culture_Deep", "prompt": "请解释中国传统文化中‘龙’的象征意义，并与西方的‘Dragon’做对比。"},
    {"id": "C63", "type": "Complex", "category": "Tech_Safety", "prompt": "在使用公共 WiFi 时，应该注意哪些安全问题？请列举三点。"},
    {"id": "C64", "type": "Complex", "category": "Emotion_Manage", "prompt": "当感到愤怒时，有哪些科学的方法可以快速平复心情？"},
    {"id": "C65", "type": "Complex", "category": "Book_Rec", "prompt": "请推荐一本关于心理学的书，并说明推荐理由。"},
    {"id": "C66", "type": "Complex", "category": "Event_Plan", "prompt": "公司要举办年会，请列出筹备工作的五个关键步骤。"},
    {"id": "C67", "type": "Complex", "category": "Logic_Fallacy", "prompt": "指出这句话的逻辑谬误：'因为大家都这么做，所以这样做是对的。'"},
    {"id": "C68", "type": "Complex", "category": "Image_Desc", "prompt": "请用文字生动地描述一幅‘日出海上’的画面。"},
    {"id": "C69", "type": "Complex", "category": "Policy_Understand", "prompt": "请通俗解释‘双减’政策的主要内容及其目的。"},
    {"id": "C70", "type": "Complex", "category": "Future_Job", "prompt": "你认为哪些职业在未来 10 年最不容易被 AI 取代？为什么？"},
    {"id": "C71", "type": "Complex", "category": "Health_Diet", "prompt": "请为减肥人士设计一份一日三餐的健康食谱。"},
    {"id": "C72", "type": "Complex", "category": "Travel_Tip", "prompt": "第一次坐飞机需要注意什么？请列出安检和登机流程。"},
    {"id": "C73", "type": "Complex", "category": "Language_Learn", "prompt": "学习外语最有效的方法有哪些？请给出三条建议。"},
    {"id": "C74", "type": "Complex", "category": "Social_Issue", "prompt": "如何看待老龄化社会带来的挑战？请提出两点应对思路。"},
    {"id": "C75", "type": "Complex", "category": "Art_Appreciate", "prompt": "请简述梵高的《星月夜》这幅画的特点。"},
    {"id": "C76", "type": "Complex", "category": "Conflict_Res", "prompt": "同事之间因为工作分配不均产生矛盾，作为旁观者该如何调解？"},
    {"id": "C77", "type": "Complex", "category": "Science_Exp", "prompt": "请描述一个简单的家庭科学实验，证明空气占据空间。"},
    {"id": "C78", "type": "Complex", "category": "News_Comment", "prompt": "针对‘网络暴力’现象，请发表一篇简短的评论，呼吁文明上网。"},
    {"id": "C79", "type": "Complex", "category": "Memory_Tech", "prompt": "有哪些提高记忆力的技巧？请介绍‘记忆宫殿’法。"},
    {"id": "C80", "type": "Complex", "category": "Gift_Select", "prompt": "要给一位喜欢读书的长辈选生日礼物，你有什么建议？请推荐三样。"},
    {"id": "C81", "type": "Complex", "category": "Time_Manage", "prompt": "请介绍‘番茄工作法’的原理和操作步骤。"},
    {"id": "C82", "type": "Complex", "category": "Env_Action", "prompt": "作为个人，我们可以为保护生物多样性做些什么？请列举三件小事。"},
    {"id": "C83", "type": "Complex", "category": "Finance_Basic", "prompt": "什么是复利？请用简单的例子说明它的威力。"},
    {"id": "C84", "type": "Complex", "category": "Pet_Care", "prompt": "养一只小狗需要注意哪些事项？请从饮食、运动、疫苗三方面回答。"},
    {"id": "C85", "type": "Complex", "category": "Movie_Rec", "prompt": "请推荐一部感人的中国电影，并简述剧情（不要剧透结局）。"},
    {"id": "C86", "type": "Complex", "category": "Habits", "prompt": "如何养成早起的好习惯？请给出可执行的步骤。"},
    {"id": "C87", "type": "Complex", "category": "Comm_Skill", "prompt": "在公开演讲时紧张怎么办？请提供三个缓解紧张的技巧。"},
    {"id": "C88", "type": "Complex", "category": "Tech_Ethic", "prompt": "AI 换脸技术可能带来哪些伦理风险？请简述。"},
    {"id": "C89", "type": "Complex", "category": "Garden", "prompt": "如何在阳台上种植薄荷？请给出种植指南。"},
    {"id": "C90", "type": "Complex", "category": "Photo_Tip", "prompt": "手机摄影有哪些构图技巧？请介绍‘三分法’。"},
    {"id": "C91", "type": "Complex", "category": "Emergency", "prompt": "遇到有人溺水，正确的急救步骤是什么？"},
    {"id": "C92", "type": "Complex", "category": "Study_Plan", "prompt": "如何准备考研英语？请制定一个半年的复习规划大纲。"},
    {"id": "C93", "type": "Complex", "category": "Relation", "prompt": "如何处理好婆媳关系？请给出三条建议。"},
    {"id": "C94", "type": "Complex", "category": "Market", "prompt": "为什么奢侈品价格那么高还有人买？请从心理学角度分析。"},
    {"id": "C95", "type": "Complex", "category": "Sleep", "prompt": "睡眠质量差有哪些改善方法？请排除药物干预。"},
    {"id": "C96", "type": "Complex", "category": "Team_Work", "prompt": "一个高效的团队需要具备哪些要素？请列举四点。"},
    {"id": "C97", "type": "Complex", "category": "History_Person", "prompt": "请评价爱因斯坦对现代物理学的贡献。"},
    {"id": "C98", "type": "Complex", "category": "Lang_Evo", "prompt": "网络流行语（如 YYDS）对传统汉语的发展是好事还是坏事？请辩证分析。"},
    {"id": "C99", "type": "Complex", "category": "Creativity", "prompt": "请为一个环保组织设计一个口号，要求朗朗上口且有感染力。"},
    {"id": "C100", "type": "Complex", "category": "Final_Thought", "prompt": "如果用一句话总结人类文明的过去，你会说什么？"}
]

# ================= 全局状态 =================
peak_mem_during_task = 0.0
throttle_event_detected = False 

# ================= 硬件读取 =================
def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except: return 0.0

def get_throttled_status():
    """
    返回: (raw_hex_string, current_low_16_bits_int, is_throttled_bool)
    只关注低 16 位，忽略历史高位。
    """
    try:
        result = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and 'throttled=' in result.stdout:
            hex_str = result.stdout.split('=')[1].strip()
            full_val = int(hex_str, 16)
            
            # 关键：只取低 16 位
            current_val = full_val & 0xFFFF
            
            # 检查 Bit 1, 2, 3
            is_throttled = bool(current_val & THROTTLE_MASK_CURRENT)
            return hex_str, current_val, is_throttled
    except Exception as e:
        print(f"Warning: Failed to get throttled status: {e}")
    return "0x0", 0, False

def get_mem_used_mb():
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(':')
                data[key] = int(parts[1]) / 1024.0
        total = data.get('MemTotal', 0)
        avail = data.get('MemAvailable', data.get('MemFree', 0) + data.get('Buffers', 0) + data.get('Cached', 0))
        return total - avail
    except: return 0.0

# ================= 初始状态 =================
def capture_initial_state():
    hex_s, val, is_throttled = get_throttled_status()
    state = {
        "timestamp_iso": datetime.now().isoformat(),
        "cpu_temp_c": get_cpu_temp(),
        "throttled_status_hex": hex_s,
        "throttled_status_low16": val,
        "is_currently_throttled": is_throttled,
        "ram_used_mb": get_mem_used_mb(),
        "model_name": LLM_MODEL,
        "config": {
            "cooldown_target_temp": TEMP_COOLDOWN_TARGET,
            "throttle_mask_hex": hex(THROTTLE_MASK_CURRENT),
            "target_samples": MAX_SAMPLES
        }
    }
    with open(FILE_INITIAL_STATE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    status_msg = "✅ 完美初始状态 (无节流历史)" if val == 0 else "⚠️ 存在历史节流记录 (但当前可能正常)"
    print(f"📸 初始状态已保存 -> {FILE_INITIAL_STATE}")
    print(f"   [T={state['cpu_temp_c']:.1f}°C, Status={hex_s}] {status_msg}")
    return state

# ================= 线程：负载中监控 =================
def monitor_load_loop(start_flag, end_flag):
    global peak_mem_during_task, throttle_event_detected
    
    peak_mem_during_task = 0.0
    throttle_event_detected = False
    
    while not end_flag.is_set():
        if start_flag.is_set():
            mem_val = get_mem_used_mb()
            if mem_val > peak_mem_during_task:
                peak_mem_during_task = mem_val
            
            _, _, is_now_throttled = get_throttled_status()
            if is_now_throttled:
                throttle_event_detected = True
                
        time.sleep(POLL_INTERVAL)

# ================= 冷却逻辑 =================
def cool_if_throttled_detected():
    if not throttle_event_detected:
        return False
    
    current_temp = get_cpu_temp()
    print(f"\n⚠️  【内核检测】发现热节流标志 (Throttled Flag Set)!")
    print(f"⏳  触发主动散热等待，目标温度 <= {TEMP_COOLDOWN_TARGET}°C...")
    
    while True:
        time.sleep(POLL_INTERVAL)
        t = get_cpu_temp()
        _, val, still_throttled = get_throttled_status()
        
        # 显示状态：温度 + 低 16 位值
        print(f"   [冷却中] T={t:.1f}°C | Flag_Val={val} ({'Active' if still_throttled else 'Cleared'})", end='\r')
        
        if t <= TEMP_COOLDOWN_TARGET and not still_throttled:
            print(f"\n✅ 系统恢复：T={t:.1f}°C, 节流标志已清除 (0x0)。继续测试。")
            return True
    
    return False

# ================= 推理执行 =================
def run_inference_raw(prompt):
    global throttle_event_detected
    
    mem_start_flag = threading.Event()
    mem_end_flag = threading.Event()
    
    throttle_event_detected = False 
    
    t_mon = threading.Thread(target=monitor_load_loop, args=(mem_start_flag, mem_end_flag))
    t_mon.daemon = True
    t_mon.start()
    
    start_req_time = time.time()
    first_token_time = None
    last_token_time = None
    full_response = ""
    token_count = 0
    error_msg = ""
    success = False
    
    mem_start_flag.set()
    
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.7}
        }, timeout=600, stream=True)
        
        if resp.status_code != 200:
            error_msg = f"HTTP {resp.status_code}"
            mem_end_flag.set()
            return {"success": False, "error": error_msg, "response": "", "ttft_ms": 0, "tps": 0, "tokens": 0, "duration_s": 0, "peak_mem_mb": 0, "throttled_flag": False}

        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    now = time.time()
                    if data.get('response'):
                        if first_token_time is None:
                            first_token_time = now
                        full_response += data['response']
                    if data.get('done'):
                        last_token_time = now
                        token_count = data.get('eval_count', 0)
                        success = True
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        error_msg = str(e)
        success = False
    finally:
        mem_end_flag.set()
        time.sleep(0.1)
        recorded_peak_mem = peak_mem_during_task
        recorded_throttled = throttle_event_detected

    ttft_ms = 0.0
    tps = 0.0
    duration_s = 0.0
    
    if first_token_time:
        ttft_ms = (first_token_time - start_req_time) * 1000.0
    
    if last_token_time and first_token_time:
        generation_time = last_token_time - first_token_time
        duration_s = last_token_time - start_req_time
        if generation_time > 0:
            tps = float(token_count) / generation_time
    elif last_token_time:
        duration_s = last_token_time - start_req_time

    return {
        "success": success, "error": error_msg, "response": full_response, 
        "tokens": token_count, "ttft_ms": ttft_ms, "tps": tps, 
        "duration_s": duration_s, "peak_mem_mb": recorded_peak_mem, 
        "throttled_flag": recorded_throttled
    }

# ================= 主程序 =================
def main():
    print("="*70)
    print("🚀 场景一：内核节流标志监测测试 (最终确认版)")
    print("="*70)
    print(f"📦 模型：{LLM_MODEL}")
    print(f"🎯 策略：监听 vcgencmd get_throttled (仅低 16 位)")
    print(f"📝 目标样本数：{MAX_SAMPLES}")
    print("-" * 70)

    capture_initial_state()
    print(f"⏳ 系统稳定等待 {COOLDOWN_START} 秒...")
    time.sleep(COOLDOWN_START)

    with open(FILE_METRICS_RAW, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Sample_ID', 'Category', 'Type', 'Prompt_Length',
            'Timestamp_ISO', 
            'TTFT_ms', 'TPS', 'Tokens', 'Duration_s', 'Peak_Mem_MB',
            'Throttled_Flag', 'Waited_For_Cooling',
            'Success', 'Error_Msg'
        ])

    content_list = []
    events_list = []
    
    sample_count = 0
    dataset_idx = 0

    print("\n开始采集数据...\n")

    try:
        while sample_count < MAX_SAMPLES:
            sample = DATASET[dataset_idx % len(DATASET)]
            dataset_idx += 1
            
            waited = cool_if_throttled_detected()
            
            if waited:
                events_list.append({
                    "event_type": "KERNEL_THROTTLE_RECOVERY",
                    "timestamp": datetime.now().isoformat(),
                    "sample_id_next": sample['id']
                })
            
            temp_start = get_cpu_temp()
            hex_s, val, _ = get_throttled_status()
            
            res = run_inference_raw(sample['prompt'])
            
            timestamp_iso = datetime.now().isoformat()
            
            with open(FILE_METRICS_RAW, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    sample['id'], sample['category'], sample['type'], len(sample['prompt']),
                    timestamp_iso,
                    f"{res['ttft_ms']:.4f}", f"{res['tps']:.4f}", res['tokens'], f"{res['duration_s']:.4f}", f"{res['peak_mem_mb']:.2f}",
                    res['throttled_flag'], waited,
                    res['success'], res['error']
                ])
            
            content_list.append({
                "sample_id": sample['id'], "category": sample['category'], "type": sample['type'],
                "prompt": sample['prompt'], "response": res['response'],
                "metrics": {
                    "ttft_ms": res['ttft_ms'], "tps": res['tps'], "tokens": res['tokens'], 
                    "duration_s": res['duration_s'], "peak_mem_mb": res['peak_mem_mb'],
                    "kernel_throttled_flag": res['throttled_flag']
                },
                "environment": {
                    "timestamp": timestamp_iso, 
                    "temp_start_c": temp_start, 
                    "initial_throttle_status_hex": hex_s,
                    "waited_for_cooling": waited
                },
                "status": {"success": res['success'], "error": res['error']},
                "human_evaluation": {"semantic_score": None, "logic_score": None, "hallucination_flag": None, "completeness_check": None, "notes": ""}
            })
            
            status = "OK" if res['success'] else "FAIL"
            throttle_status = "⚠️THROTTLED" if res['throttled_flag'] else "✅Clean"
            wait_indicator = "⏳" if waited else "  "
            
            print(f"[{sample_count+1:02d}/{MAX_SAMPLES}] {wait_indicator} {sample['category']:10s} | "
                  f"TTFT:{res['ttft_ms']:7.1f}ms | TPS:{res['tps']:6.2f} | "
                  f"Mem:{res['peak_mem_mb']:6.1f}MB | {throttle_status:12s} | {status}")
            
            sample_count += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n⛔ 用户中断")
        events_list.append({"event_type": "USER_INTERRUPT", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        print(f"\n💥 错误：{e}")
        events_list.append({"event_type": "SYSTEM_ERROR", "timestamp": datetime.now().isoformat(), "msg": str(e)})
    finally:
        with open(FILE_CONTENT_RAW, 'w', encoding='utf-8') as f:
            json.dump(content_list, f, ensure_ascii=False, indent=2)
        with open(FILE_EVENTS_RAW, 'w', encoding='utf-8') as f:
            json.dump(events_list, f, ensure_ascii=False, indent=2)

        print("\n" + "="*70)
        print("✅ 数据采集完成")
        print(f"📂 文件列表:")
        print(f"   - {FILE_METRICS_RAW}")
        print(f"   - {FILE_CONTENT_RAW}")
        print("="*70)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ pip install requests")
        sys.exit(1)
    main()