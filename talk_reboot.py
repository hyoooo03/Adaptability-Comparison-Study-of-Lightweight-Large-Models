import requests
import os
import serial
import time
import sys
import subprocess
import re
import csv
import pandas as pd
from datetime import datetime
from gpiozero import Button

# ================= 配置区域 =================
AUDIO_FILE = "/home/qhyoooo/code/test/test1.wav"
RECORD_DURATION = 5       # 录音时长
BUTTON_PIN = 21           # 引脚确认

# 数据记录文件名
CSV_FILENAME = f"e2e_test_data_{datetime.now().strftime('%H%M%S')}.csv"

# ⚠️ 自动检测设备
def find_usb_audio_device():
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if "USB" in line or "Device" in line:
                match = re.search(r'card (\d+):.*device (\d+)', line)
                if match:
                    card = match.group(1)
                    device = match.group(2)
                    dev_str = f"hw:{card},{device}"
                    print(f"🔍 自动发现 USB 声卡: {dev_str}")
                    return dev_str
        print("⚠️ 未找到 USB 声卡，默认使用 hw:0,0")
        return "hw:0,0"
    except Exception as e:
        print(f"⚠️ 检测设备失败: {e}，默认使用 hw:0,0")
        return "hw:0,0"

AUDIO_DEVICE = find_usb_audio_device()

# ASR / LLM / TTS 配置
ASR_URL = "http://localhost:8888/v1/audio/transcriptions"
ASR_API_KEY = "sk-........................"
ASR_MODEL = "qwen3-asr-flash"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "Qwen3:1.7b"
TTS_PORT = '/dev/serial0'
TTS_BAUDRATE = 9600
# ===========================================

# 数据记录列表
data_rows = []

def init_serial():
    try:
        ser = serial.Serial(port=TTS_PORT, baudrate=TTS_BAUDRATE, timeout=1)
        time.sleep(0.5)
        print(f"✅ TTS 串口 ({TTS_PORT}) 打开成功")
        return ser
    except Exception as e:
        print(f"❌ 串口失败: {e}")
        return None

def speak_text(ser, text):
    if not ser or not text: return
    print(f"🔊 播报: {text}")
    try:
        ser.write(f"<G>{text}".encode('gbk'))
        time.sleep(max(1.0, len(text) * 0.3))
    except Exception as e:
        print(f"❌ 播报错: {e}")

def record_audio(output_path):
    print(f"🎤 录音中... ({RECORD_DURATION}秒) [设备: {AUDIO_DEVICE}]")
    
    cmd = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", "cd",
        "-c", "1",
        "-d", str(RECORD_DURATION),
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 录音保存完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 录音失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ 未找到 arecord 命令，请运行: sudo apt install alsa-utils")
        return False

def speech_to_text(file_path):
    if not os.path.exists(file_path): return None, None, None
    
    # 【新增】记录 ASR 开始时间
    t_asr_start = time.time()
    print("📡 识别中...")
    
    try:
        with open(file_path, 'rb') as f:
            resp = requests.post(ASR_URL, headers={"Authorization": f"Bearer {ASR_API_KEY}"}, 
                                 data={"model": ASR_MODEL}, files={'file': f}, timeout=30)
        
        # 【新增】记录 ASR 结束时间
        t_asr_end = time.time()
        
        if resp.status_code == 200:
            text = resp.json().get('text', '').strip()
            if text: 
                print(f"👂 结果: {text}")
                # 【新增】返回文本和起止时间
                return text, t_asr_start, t_asr_end
    except Exception as e:
        print(f"❌ ASR 错: {e}")
        return None, None, None
        
    return None, None, None

def ask_llama(prompt):
    # 【新增】记录 LLM 开始时间
    t_llm_start = time.time()
    print("🧠 思考中...")
    
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": f"简短回答：{prompt}",
            "stream": False
        }, timeout=120)
        
        # 【新增】记录 LLM 结束时间
        t_llm_end = time.time()
        
        if resp.status_code == 200:
            ans = resp.json().get('response', '').strip().replace('"','')
            # 【新增】返回回答和起止时间
            return ans, t_llm_start, t_llm_end
    except Exception as e:
        print(f"❌ LLM 错: {e}")
        return None, None, None
        
    return None, None, None

def wait_for_button_press_custom(button):
    while not button.is_pressed:
        time.sleep(0.1)
    while button.is_pressed:
        time.sleep(0.1)
    print("🔘 检测到按下动作！")

def wait_for_button_release_custom(button):
    while not button.is_pressed:
        time.sleep(0.1)
    print("⏸️ 按钮已松开")

def main():
    global data_rows
    print("🤖 智能助手启动 (带时间戳记录版)")
    ser = init_serial()
    if ser: speak_text(ser, "系统就绪，请按按钮说话。")

    try:
        # 必须开启上拉电阻，防止悬空误触发
        button = Button(BUTTON_PIN, pull_up=True)
        print(f"✅ 监听 GPIO {BUTTON_PIN} ...")
    except Exception as e:
        print(f"❌ 按键初始化失败: {e}")
        return

    # 测试轮次控制，防止无限运行，这里设为20轮
    MAX_ROUNDS = 20
    round_count = 0

    while round_count < MAX_ROUNDS:
        round_count += 1
        print(f"\n--- 第 {round_count}/{MAX_ROUNDS} 轮 ---")
        print("⏳ 等待按键开始...")
        
        # 1. 等待按键按下
        wait_for_button_press_custom(button)
        
        # 【打点1】记录开始时间 T_start
        t_start = time.time()
        print(f"⏱️ [T_start] 录音开始: {t_start:.6f}")
        
        # 2. 执行流程
        if record_audio(AUDIO_FILE):
            # 【修改】接收 ASR 返回的时间和文本
            text, t_asr_start, t_asr_end = speech_to_text(AUDIO_FILE)
            
            if text:
                # 【修改】接收 LLM 返回的时间和回答
                ans, t_llm_start, t_llm_end = ask_llama(text)
                
                if ans:
                    # 【新增】记录 TTS 发送开始时间
                    t_tts_start = time.time()
                    
                    speak_text(ser, ans)
                    
                    # 【打点2】播报结束后，等待用户人为确认结束
                    print("🔊 播报完毕，请在听到声音结束后，再次按下按键标记结束。")
                    wait_for_button_press_custom(button)
                    
                    # 【打点3】记录结束时间 T_end
                    t_end = time.time()
                    print(f"⏹️ [T_end] 用户确认结束: {t_end:.6f}")
                    
                    # 保存数据 (只记录原始时间戳，不做计算)
                    data_rows.append({
                        "Round": round_count,
                        "Question": text,
                        "Answer": ans,
                        "T_Start": t_start,
                        "T_ASR_Start": t_asr_start,
                        "T_ASR_End": t_asr_end,
                        "T_LLM_Start": t_llm_start,
                        "T_LLM_End": t_llm_end,
                        "T_TTS_Start": t_tts_start,
                        "T_End": t_end
                    })
                    
                else:
                    speak_text(ser, "我没听明白。")
            else:
                speak_text(ser, "没听清，请重试。")
        else:
            speak_text(ser, "录音失败。")
        
        # 等待松开，进入下一轮
        wait_for_button_release_custom(button)
        time.sleep(0.5)

    # 循环结束，保存数据
    df = pd.DataFrame(data_rows)
    df.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
    print(f"\n📊 测试完成！数据已保存至: {CSV_FILENAME}")

if __name__ == "__main__":
    try:
        import requests, serial, gpiozero
    except ImportError as e:
        print(f"❌ 缺库: {e}")
        sys.exit(1)
    
    main()