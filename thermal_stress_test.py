import requests
import time
import sys
import subprocess
import csv
import json
import re
import threading
from datetime import datetime
import os

# ================= 配置区域 =================
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:1b"

TEST_DURATION_MIN = 15          # 建议 15 分钟以覆盖完整热演化过程
COOLDOWN_START = 10             
TEMP_AMBIENT_MAX = 45.0         # 对应文档要求的 45 度以下

# 硬件参数 (仅用于记录参考，不再用于判断逻辑)
CPU_FREQ_NOMINAL_MHZ = 2400     
FREQ_THROTTLE_LIMIT_MHZ = CPU_FREQ_NOMINAL_MHZ * 0.95 

# 节流标志位掩码 (只关注低 16 位的 Bit 1, 2, 3)
# Bit 1 (0x2): Arm frequency capped
# Bit 2 (0x4): Currently throttled
# Bit 3 (0x8): Soft temperature limit active
THROTTLE_MASK_CURRENT = 0x000E 

# 输出文件
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_INITIAL_STATE = f"raw_initial_state_{TIMESTAMP}.json"
RAW_THERMAL_TRACE = f"raw_thermal_trace_{TIMESTAMP}.csv"
RAW_INFERENCE_LOG = f"raw_inference_log_{TIMESTAMP}.csv"
RAW_FULL_RESPONSES = f"raw_full_responses_{TIMESTAMP}.json"
RAW_EVENTS_LOG = f"raw_events_{TIMESTAMP}.json"

# 压力测试 Prompt (长文本)
STRESS_PROMPT = """
请撰写一篇关于"边缘计算设备在无风扇被动散热条件下的热管理挑战与优化策略"的技术综述。
要求：字数不少于 800 字，结构清晰，包含引言、热产生机制、被动散热局限性、DVFS影响及未来展望。直接开始正文。
"""
DATASET = [{"type": "Stress_LongGen", "prompt": STRESS_PROMPT}]

# ================= 全局状态 =================
stop_monitoring = False
start_total_time = 0

# ================= 硬件读取函数 =================
def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except: return 0.0

def get_cpu_freq():
    try:
        # 尝试 vcgencmd (树莓派)
        result = subprocess.run(['vcgencmd', 'measure_clock', 'arm'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            match = re.search(r'frequency\(\d+\)=(\d+)', result.stdout)
            if match: return int(match.group(1)) / 1000000.0
        
        # 备选：通用 Linux 方法
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except: pass
    return 0.0

def get_throttled_status():
    """
    获取树莓派节流状态。
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
        # print(f"Warning: Failed to get throttled status: {e}")
        pass
    return "0x0", 0, False

def get_system_mem_info():
    """获取详细的系统内存信息"""
    mem_info = {
        "total_mb": 0, "used_mb": 0, "free_mb": 0,
        "shared_mb": 0, "buff_cache_mb": 0, "available_mb": 0
    }
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(':')
                data[key] = int(parts[1])
        
        mem_info['total_mb'] = data.get('MemTotal', 0) / 1024.0
        free_kb = data.get('MemFree', 0)
        buff_kb = data.get('Buffers', 0)
        cache_kb = data.get('Cached', 0)
        avail_kb = data.get('MemAvailable', free_kb)
        
        mem_info['free_mb'] = free_kb / 1024.0
        mem_info['buff_cache_mb'] = (buff_kb + cache_kb) / 1024.0
        mem_info['available_mb'] = avail_kb / 1024.0
        mem_info['used_mb'] = mem_info['total_mb'] - mem_info['available_mb']
        mem_info['shared_mb'] = data.get('Shmem', 0) / 1024.0
    except Exception as e:
        pass
    return mem_info

def get_gpu_mem_specific():
    """尝试获取具体的显存占用"""
    gpu_info = {"allocated_mb": 0, "method": "unknown"}
    try:
        result = subprocess.run(['vcgencmd', 'get_mem', 'gpu'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and 'gpu' in result.stdout:
            match = re.search(r'gpu=(\d+)', result.stdout)
            if match:
                gpu_info['allocated_mb'] = int(match.group(1))
                gpu_info['method'] = "vcgencmd_get_mem"
                return gpu_info
    except: pass
    
    mem_info = get_system_mem_info()
    gpu_info['allocated_mb'] = mem_info['shared_mb']
    gpu_info['method'] = "shmem_approximation"
    return gpu_info

# ================= 初始状态捕获函数 =================
def capture_initial_state():
    print("📸 正在捕获初始设备基准状态...")
    
    temp = get_cpu_temp()
    freq = get_cpu_freq()
    hex_s, val, is_throttled = get_throttled_status() # [新增] 获取节流状态
    mem_info = get_system_mem_info()
    gpu_info = get_gpu_mem_specific()
    
    uptime_sec = 0
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_sec = float(f.readline().split()[0])
    except: pass
    
    initial_data = {
        "timestamp_iso": datetime.now().isoformat(),
        "system_uptime_sec": uptime_sec,
        "thermal": {"cpu_temp_c": temp},
        "frequency": {
            "cpu_freq_mhz": freq,
            "nominal_limit_mhz": CPU_FREQ_NOMINAL_MHZ,
            "throttle_limit_mhz": FREQ_THROTTLE_LIMIT_MHZ
        },
        "throttle_status": { # [新增] 记录初始节流状态
            "raw_hex": hex_s,
            "low_16_bits": val,
            "is_throttled": is_throttled,
            "mask_used": hex(THROTTLE_MASK_CURRENT)
        },
        "memory": {
            "total_mb": mem_info['total_mb'],
            "used_mb": mem_info['used_mb'],
            "free_mb": mem_info['free_mb'],
            "available_mb": mem_info['available_mb'],
            "buff_cache_mb": mem_info['buff_cache_mb'],
            "shared_mb": mem_info['shared_mb']
        },
        "gpu_estimate": {
            "estimated_vram_usage_mb": gpu_info['allocated_mb'],
            "detection_method": gpu_info['method']
        },
        "environment": {
            "ambient_threshold_c": TEMP_AMBIENT_MAX,
            "test_duration_min": TEST_DURATION_MIN
        }
    }
    
    try:
        with open(RAW_INITIAL_STATE, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 初始状态已保存至: {RAW_INITIAL_STATE}")
        
        status_msg = "✅ 完美初始状态 (无节流)" if (val == 0 and not is_throttled) else "⚠️ 初始已存在节流标志"
        print(f"   [初始状态摘要]")
        print(f"   🌡️  温度: {temp:.2f}°C")
        print(f"   ⚡ 频率: {freq:.1f} MHz")
        print(f"   🛑 节流标志: {hex_s} ({status_msg})")
        print(f"   💾 内存占用: {mem_info['used_mb']:.1f} / {mem_info['total_mb']:.1f} MB")
        print(f"   🎮 显存/共享估算: {gpu_info['allocated_mb']:.1f} MB")
        
        return initial_data
    except Exception as e:
        print(f"❌ 保存初始状态失败: {e}")
        return None

# ================= 后台监控线程 (5Hz) =================
def monitor_thread_func(start_time_ref, trace_file):
    global stop_monitoring
    print(f"📡 [后台] 监控线程启动 (5Hz: Temp, Freq, Mem, Throttle)...")
    
    with open(trace_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # [修改] 表头增加 Throttle_Flag
        writer.writerow(['Elapsed_Seconds', 'Temp_C', 'Freq_MHz', 'Mem_Used_MB', 'Throttle_Flag'])
        
        while not stop_monitoring:
            loop_start = time.time()
            temp = get_cpu_temp()
            freq = get_cpu_freq()
            mem = get_system_mem_info()['used_mb']
            _, _, is_throttled = get_throttled_status() # [修改] 获取节流状态
            
            elapsed_sec = time.time() - start_time_ref
            
            writer.writerow([
                f"{elapsed_sec:.4f}",
                f"{temp:.2f}",
                f"{freq:.1f}",
                f"{mem:.1f}",
                1 if is_throttled else 0 # [修改] 记录标志
            ])
            f.flush()
            
            sleep_time = 0.2 - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

# ================= 推理执行函数 =================
def run_inference(prompt):
    start_time = time.time()
    first_token_time = None
    full_response = ""
    token_count = 0
    throttle_detected_during_task = False # [新增] 任务内节流标记
    
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": -1, "temperature": 0.7}
        }, timeout=600, stream=True)
        
        if resp.status_code != 200:
            return "", 0, 0, 0, False, 0, False

        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    now = time.time()
                    
                    # [新增] 在流式接收过程中持续检查节流状态
                    _, _, is_now_throttled = get_throttled_status()
                    if is_now_throttled:
                        throttle_detected_during_task = True
                    
                    if first_token_time is None and data.get('response'):
                        first_token_time = now
                    
                    if data.get('response'):
                        full_response += data['response']
                    
                    if data.get('done'):
                        end_time = now
                        duration = end_time - start_time
                        ttft = (first_token_time - start_time) if first_token_time else 0
                        token_count = data.get('eval_count', len(full_response))
                        
                        generation_time = end_time - first_token_time if first_token_time else duration
                        tps = token_count / generation_time if generation_time > 0 else 0
                        peak_mem = get_system_mem_info()['used_mb']
                        
                        # [修改] 返回节流标记
                        return full_response, ttft, tps, token_count, True, peak_mem, throttle_detected_during_task
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return "", 0, 0, 0, False, 0, False
    
    return "", 0, 0, 0, False, 0, False

# ================= 主程序 =================
def main():
    global stop_monitoring, start_total_time

    print(f"🚀 [最终版+] 场景二：非稳态热演化压力测试 (内核节流监测)")
    print(f"📦 模型: {LLM_MODEL}")
    print(f"⏱️  时长: {TEST_DURATION_MIN} 分钟")
    print(f"🛡️  检测: vcgencmd get_throttled (内核级标志)")
    
    # 1. 捕获并保存初始状态
    initial_state = capture_initial_state()
    if not initial_state:
        print("⚠️  未能保存初始状态，但将继续测试。")
    
    # 2. 初始检查
    initial_temp = initial_state['thermal']['cpu_temp_c'] if initial_state else get_cpu_temp()
    initial_throttled = initial_state['throttle_status']['is_throttled'] if initial_state else False
    
    print(f"🌡️  初始温度确认: {initial_temp:.1f}°C (阈值:{TEMP_AMBIENT_MAX}°C)")
    if initial_temp > TEMP_AMBIENT_MAX:
        print(f"⚠️  警告: 初始温度过高。")
        if input("强行继续？(y/n): ").lower() != 'y': sys.exit(0)
    
    if initial_throttled:
        print(f"⚠️  警告: 初始已检测到节流标志！数据可能受之前运行影响。")
        if input("强行继续？(y/n): ").lower() != 'y': sys.exit(0)
    else:
        print("✅ 初始状态符合实验要求 (无节流)。")
    
    time.sleep(COOLDOWN_START)
    print("⏳ 稳定等待结束，开始记录...")

    # 3. 初始化文件
    with open(RAW_INFERENCE_LOG, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # [修改] 表头增加 Throttled_Flag
        writer.writerow([
            'Task_ID', 
            'Start_Time_ISO', 
            'Elapsed_Min', 
            'TTFT_s', 
            'TPS', 
            'Tokens', 
            'Duration_s', 
            'Temp_End_C', 
            'Freq_End_MHz', 
            'Mem_Peak_MB', 
            'Throttled_Flag', # [新增]
            'Success'
        ])

    events_list = []
    responses_list = []
    
    # 4. 启动监控
    start_total_time = time.time()
    t_mon = threading.Thread(target=monitor_thread_func, args=(start_total_time, RAW_THERMAL_TRACE))
    t_mon.daemon = True
    t_mon.start()

    end_time = start_total_time + (TEST_DURATION_MIN * 60)
    task_id = 0
    throttle_first_detected = False # 全局首次检测标记

    print("-" * 50)
    try:
        while time.time() < end_time:
            prompt = DATASET[0]['prompt']
            
            # 执行推理
            resp_text, ttft, tps, tokens, success, peak_mem, is_throttled = run_inference(prompt)
            
            # 采集结束时刻状态
            t_end = time.time()
            temp = get_cpu_temp()
            freq = get_cpu_freq()
            elapsed_min = (t_end - start_total_time) / 60.0
            
            # [修改] 使用内核标志判断状态
            status = "OK" if success else "FAIL"
            if is_throttled:
                status = "THROTTLED"
                if not throttle_first_detected:
                    throttle_first_detected = True
                    events_list.append({
                        'type': 'KERNEL_THROTTLE_FIRST', 
                        'time_min': elapsed_min, 
                        'freq': freq, 
                        'temp': temp,
                        'note': 'Detected via vcgencmd get_throttled flags'
                    })
                    print(f"\n⚠️  [{elapsed_min:.2f}m] 【内核级检测】首次触发节流标志! (T={temp:.1f}°C, F={freq:.1f}MHz)")

            # 控制台简略输出
            throttle_indicator = "⚠️" if is_throttled else "  "
            print(f"[{elapsed_min:.1f}m] {throttle_indicator} T:{temp:.1f}°C | F:{freq:.1f}MHz | M:{peak_mem:.0f}MB | TPS:{tps:.1f} | {status}")

            # 写入 CSV
            with open(RAW_INFERENCE_LOG, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    task_id,
                    datetime.now().isoformat(),
                    f"{elapsed_min:.4f}",
                    f"{ttft:.3f}",
                    f"{tps:.2f}",
                    tokens,
                    f"{ttft + (tokens/tps if tps>0 else 0):.3f}",
                    f"{temp:.2f}",
                    f"{freq:.1f}",
                    f"{peak_mem:.1f}",
                    1 if is_throttled else 0, # [修改] 写入标志
                    success
                ])
            
            # 保存完整文本
            responses_list.append({
                "id": task_id,
                "time_min": elapsed_min,
                "prompt": prompt,
                "response": resp_text,
                "metrics": {
                    "temp": temp, 
                    "freq": freq, 
                    "tps": tps, 
                    "mem": peak_mem, 
                    "ttft": ttft,
                    "throttled_flag": is_throttled # [修改] 保存标志
                }
            })

            task_id += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n⛔ 用户中断")
    except Exception as e:
        print(f"\n💥 错误: {e}")
        events_list.append({'type': 'ERROR', 'msg': str(e)})
    finally:
        stop_monitoring = True
        t_mon.join(timeout=2)
        
        # 保存辅助文件
        with open(RAW_EVENTS_LOG, 'w') as f: json.dump(events_list, f, indent=2)
        with open(RAW_FULL_RESPONSES, 'w', encoding='utf-8') as f: json.dump(responses_list, f, ensure_ascii=False, indent=2)

        print("\n" + "="*50)
        print("✅ 数据采集完成")
        print(f"📋 初始基准状态 (含节流标志): {RAW_INITIAL_STATE}")
        print(f"📈 5Hz 遥测数据 (含 Throttle_Flag): {RAW_THERMAL_TRACE}")
        print(f"📊 任务级指标 (含 Throttled_Flag): {RAW_INFERENCE_LOG}")
        print(f"📝 完整文本/事件: {RAW_FULL_RESPONSES}")
        print("\n💡 数据分析提示:")
        print("   - 筛选 'Throttled_Flag' == 0 的数据作为冷机/未饱和基准。")
        print("   - 筛选 'Throttled_Flag' == 1 的数据分析热衰减曲线。")
        print("="*50)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("❌ pip install requests")
        sys.exit(1)
    main()