import os
import json
import time
import re
import pandas as pd
import requests
from openai import OpenAI
import dashscope

# ---------------------------------------------------------
# 1. API 密钥配置 (请替换为你申请的真实 Key)
# ---------------------------------------------------------
OPENAI_API_KEY = "sk-k1OG4IQFqL0tH21d8MEZUjUGBMwdFfDcY985iqKbTurxrtfy"
DEEPSEEK_API_KEY = "sk-9f03b48ba90f425a92a518e97d6a0b0b"
# 百度文心一言 API Key（从 https://console.bce.baidu.com/qianfan/ais/console/apiKey 获取）
QIANFAN_API_KEY = "bce-v3/ALTAK-MLb0SZytXCwRBf96dPY1w/6bb950f798115dc82abed7c9ccfcb71f34badf4b"

# 阿里通义千问配置
dashscope.api_key = "sk-88ad1d5785c74ab59b6b532b5445c70f"

BASE_URL = "https://api.bytflux.top/v1"

# 初始化客户端
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
qianfan_client = OpenAI(base_url='https://qianfan.baidubce.com/v2', api_key=QIANFAN_API_KEY)

# ---------------------------------------------------------
# 2. 定义提示词 (强注入阶段：题库外少样本示例 Few-Shot)
# ---------------------------------------------------------
SYSTEM_PROMPT_STRONG = """你是一个深谙中国社会与高语境文化的大语言模型助手。请基于“高语境沟通”、“权力距离”、“面子”和“差序格局”等核心文化概念，客观、准确地分析社交场景。

为了帮你更好地理解推理逻辑，以下是两个标准分析示例（Few-Shot Examples）：

【示例 1：职场潜台词 (Indirect_Request)】
场景：跨部门会议上，隔壁部门经理对你说：“你们部门上次那个数据分析模型做得真漂亮，要是我们也能看懂就好了。”
输出：
{
  "intent": "暗示你用该模型帮他们部门处理数据，或者向你无偿索要模型代码。",
  "action": "希望你主动提出帮他们跑数据或分享核心模型。"
}

【示例 2：社交冒犯 (Social_Faux_Pas)】
场景：在领导精心准备的庆功宴上，领导刚端起酒杯准备致辞，新员工小李突然站起来大声说：“今天大家吃好喝好，这顿我来买单！”
输出：
{
  "has_error": "是",
  "reason": "喧宾夺主。在领导的主场抢夺买单权，破坏了领导通过请客施恩和建立权威的意图。",
  "psychological_harm": "让领导觉得被当众驳了面子，权威受到严重挑战，感到极度恼火和难堪。"
}

现在，请你使用相同的文化逻辑和严格的 JSON 格式，分析用户提供的场景。不要包含任何额外的 markdown 标记或解释。"""

def build_user_prompt(scenario_text, category):
    if category == "Indirect_Request":
        return f"""
场景：{scenario_text}

请分析上述场景，并严格以 JSON 格式返回以下字段：
{{
  "intent": "说话人的真实意图是什么？",
  "action": "他希望听话人接下来采取什么行动？"
}}
"""
    elif category == "Social_Faux_Pas":
        return f"""
场景：{scenario_text}

请分析上述场景，并严格以 JSON 格式返回以下字段：
{{
  "has_error": "这个场景中是否有人说错话或做错事？(是/否)",
  "reason": "为什么这种行为不合适？",
  "psychological_harm": "这会对他人造成怎样的心理伤害？"
}}
"""
    return ""

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"raw_text_error": text}

# ---------------------------------------------------------
# 3. 核心路由函数：全局加入自动重试机制
# ---------------------------------------------------------
def call_model_api(model_name, system_prompt, user_prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. GPT-4o
            if model_name == "gpt-4o":
                url = f"{BASE_URL}/chat/completions"
                headers = {
                    'Accept': 'application/json',
                    'Authorization': OPENAI_API_KEY,
                    'Content-Type': 'application/json'
                }
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"}
                }
                response = requests.post(url, headers=headers, json=payload, timeout=40)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

            # 2. DeepSeek
            elif model_name == "deepseek-chat":
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    stream=False,
                    extra_body={"thinking": {"type": "enabled"}},
                    temperature=0.3
                )
                return response.choices[0].message.content

            # 3. 通义千问 (Qwen)
            elif model_name == "qwen-max":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = dashscope.Generation.call(
                    model=dashscope.Generation.Models.qwen_max,
                    messages=messages,
                    result_format='message',
                    temperature=0.3
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content
                else:
                    raise Exception(f"Qwen API Error: {response.code} - {response.message}")

            # 4. 文心一言 (ERNIE)
            elif model_name == "ERNIE-4.0-8K":
                response = qianfan_client.chat.completions.create(
                    model="ernie-4.0-8k",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    extra_body={
                        "penalty_score": 1,
                        "stop": [],
                        "web_search": {"enable": False, "enable_trace": False}
                    }
                )
                return response.choices[0].message.content

            else:
                return '{"error": "未知的模型名称"}'

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      [网络/并发报错] {model_name} 调用失败，等待 3 秒后尝试第 {attempt + 2} 次重试... (错误原因: {str(e)[:50]}...)")
                time.sleep(3)
            else:
                print(f"      [彻底失败] {model_name} 重试3次后依然报错。")
                return f'{{"error": "重试3次后依然失败: {str(e)}"}}'

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    csv_file_path = "scenarios.csv"
    
    try:
        df_scenarios = pd.read_csv(csv_file_path, encoding="utf-8")
        scenarios_data = df_scenarios.to_dict(orient="records")
        print(f"成功导入 {len(scenarios_data)} 条测试场景！\n")
    except FileNotFoundError:
        print("错误：找不到 scenarios.csv 文件。")
        scenarios_data = []

    models_to_test = ["gpt-4o", "deepseek-chat", "qwen-max", "ERNIE-4.0-8K"]
    results = []

    if scenarios_data:
        print("开始执行多模型基准测试 (强注入阶段: 提供解题模板与少样本示例)...")
        print("-" * 50)
        
        for item in scenarios_data:
            print(f"正在处理场景 ID: {item['id']}...")
            user_prompt = build_user_prompt(item["scenario_text"], item["category"])
            
            for model_name in models_to_test:
                print(f"  -> 请求模型: {model_name}")
                raw_response = call_model_api(model_name, SYSTEM_PROMPT_STRONG, user_prompt)
                
                parsed_json = extract_json_from_text(raw_response)
                
                record = {
                    "scenario_id": item["id"],
                    "scenario_text": item["scenario_text"],
                    "category": item["category"],
                    "model_name": model_name
                }
                record.update(parsed_json)
                results.append(record)
                
                time.sleep(1.5) 
                
        print("-" * 50)
        print("所有模型与场景处理完成，正在保存结果...")
        
        # 导出为表示强注入结果的新文件
        output_file = "all_models_strong_injection.csv"
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by=["scenario_id", "model_name"])
        df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        print(f"测试结果已成功保存至: {output_file}")
    else:
        print("没有加载到任何场景，程序退出。")