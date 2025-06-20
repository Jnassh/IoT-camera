import os
import dashscope
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime, timedelta
import logging
import re

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# API Key
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    logger.warning("API Key 未提供，使用默认密钥")
    dashscope.api_key = "sk-cc0d38bc65a447aaa86c2717a2106d37"
else:
    dashscope.api_key = api_key
    logger.debug("成功加载 API Key")

def notify_admin(message):
    logger.info(f"通知管理员：{message}")

def read_log_file(time_range=None):
    try:
        with open("log.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not time_range:
            return "\n".join(line.strip() for line in lines)

        # 解析时间范围（格式：(start_time, end_time)）
        start_time, end_time = time_range
        filtered_lines = []
        for line in lines:
            try:
                timestamp_str = line.split(", ")[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                if start_time <= timestamp <= end_time:
                    filtered_lines.append(line.strip())
            except (ValueError, IndexError) as e:
                logger.warning(f"日志行格式错误，跳过：{line.strip()}，错误：{e}")
                continue
        return "\n".join(filtered_lines) if filtered_lines else "指定时间范围内没有日志记录"
    except Exception as e:
        logger.error(f"读取 log.txt 失败：{e}")
        return f"错误：无法读取 log.txt，原因：{str(e)}"

def get_current_time():
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"当前时间：{now}")
        return now
    except Exception as e:
        logger.error(f"时间失败：{e}")
        return f"错误：无法获取当前时间，原因：{str(e)}"

def analyze_images_by_species(species_name):
    try:
        with open("log.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        matched_images = [(line.strip().split(", ")[-1], line) for line in lines if species_name.lower() in line.lower()]
        if not matched_images:
            return {"text": f"未找到 {species_name} 图像", "images": []}

        results = []
        image_base64_list = []
        for img_path, log_line in matched_images:
            if not os.path.exists(img_path):
                logger.warning(f"图像不存在：{img_path}")
                continue
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            img_url = f"data:image/jpeg;base64,{img_base64}"
            analysis_result = analyze_image(img_url)
            results.append(f"日志记录：{log_line}\n分析结果：{analysis_result}\n")
            image_base64_list.append(img_base64)
        text_result = "\n".join(results) if results else f"未能分析任何图像"
        return {"text": text_result, "images": image_base64_list}
    except Exception as e:
        logger.error(f"分析图像失败：{e}")
        return {"text": f"错误：{str(e)}", "images": []}

def call_qwen_vl(messages):
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=dashscope.api_key,
            model='qwen-vl-max',
            messages=messages
        )
        logger.debug(f"API 响应：{response}")
        if not hasattr(response, 'output') or not response.output:
            return "API 响应缺少 output"
        choice = response.output.choices[0]
        content = choice.message.content
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                return item['text']
        return "API 未找到 text"
    except Exception as e:
        logger.error(f"API 失败：{e}")
        return f"错误：{str(e)}"

def analyze_image(image_url):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": "分析图像里的野生动物，返回类型、行为模式、是否异常"}
            ]
        }
    ]
    return call_qwen_vl(messages)

def continue_conversation(user_input, conversation_history):
    # 如果对话历史为空，初始化一个默认消息
    if not conversation_history:
        conversation_history = [{"role": "system", "content": [{"text": "你是一个野生动物保护助手"}]}]

    # 直接处理"现在几点"请求
    if user_input.strip().lower() in ["现在几点", "现在几点了", "当前时间"]:
        current_time = get_current_time()
        conversation_history.append({"role": "assistant", "content": [{"text": f"当前时间是：{current_time}"}]})
        return {"text": f"当前时间是：{current_time}", "images": []}, conversation_history

    function_prompt = """
你是一个具备函数调用能力的野生动物助手，能够调用以下函数：
- read_log_file(time_range=None)：读取日志文件。如果需要按时间段读取，time_range 参数格式为 (start_time, end_time)，时间格式为 "YYYY-MM-DD HH:MM:SS"。不带时间范围时，返回所有日志。
- get_current_time()：获取当前时间，格式为 "YYYY-MM-DD HH:MM:SS"。
- analyze_images_by_species(species_name)：分析指定物种的图像。

当用户请求合适时，请直接输出 CALL: 函数名(...) 的格式：
- 如果用户提问涉及时间范围（如"昨天出现了什么动物"、"今天早上有哪些动物"），需要先调用 CALL: get_current_time() 获取当前时间，然后根据当前时间和用户输入计算时间范围，调用 CALL: read_log_file(("start_time", "end_time"))。
  - 示例：用户输入"昨天出现了什么动物"，当前时间为 "2025-05-30 06:56:00"，计算昨天时间范围为 "2025-05-29 00:00:00" 到 "2025-05-29 23:59:59"，调用 CALL: read_log_file(("2025-05-29 00:00:00", "2025-05-29 23:59:59"))。
  - 示例：用户输入"今天早上有哪些动物"，当前时间为 "2025-05-30 06:56:00"，计算今天早上时间范围为 "2025-05-30 00:00:00" 到 "2025-05-30 12:00:00"，调用 CALL: read_log_file(("2025-05-30 00:00:00", "2025-05-30 12:00:00"))。
- 如果用户请求"分析日志"，直接调用 CALL: read_log_file()，不带时间范围。
- 如果用户请求分析特定物种的图像，调用 CALL: analyze_images_by_species("species_name")。

例如：
- 用户请求"昨天出现了什么动物"，输出：CALL: get_current_time()
- 用户请求"分析日志"，输出：CALL: read_log_file()
"""

    conversation_history.append({"role": "system", "content": [{"text": function_prompt}]})
    conversation_history.append({"role": "user", "content": [{"text": user_input}]})

    result = call_qwen_vl(conversation_history)

    if "CALL: get_current_time()" in result:
        now = get_current_time()
        conversation_history.append({"role": "assistant", "content": [{"text": f"当前时间是：{now}"}]})
        conversation_history.append({"role": "user", "content": [{"text": f"当前时间是 {now}，根据这个时间计算时间范围并读取日志：{user_input}"}]})
        result = call_qwen_vl(conversation_history)

    if "CALL: read_log_file((\"20" in result:  # 匹配带时间范围的调用
        match = re.search(r'CALL: read_log_file\(\("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", "(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"\)\)', result)
        if match:
            start_time_str, end_time_str = match.group(1), match.group(2)
            try:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
                time_range = (start_time, end_time)
                log_content = read_log_file(time_range)
                conversation_history.append({"role": "assistant", "content": [{"text": f"CALL: read_log_file((\"{start_time_str}\", \"{end_time_str}\")) 返回：\n{log_content}"}]})
                conversation_history.append({"role": "user", "content": [{"text": f"分析以下日志，列出出现的动物种类\n{log_content}"}]})
                result = call_qwen_vl(conversation_history)
            except ValueError as e:
                result = f"错误：时间格式错误，{str(e)}"
        else:
            result = "错误：无法解析时间范围"
        conversation_history.append({"role": "assistant", "content": [{"text": result}]})
        return {"text": result, "images": []}, conversation_history

    if "CALL: read_log_file()" in result:
        log_content = read_log_file()  # 不带时间范围
        conversation_history.append({"role": "assistant", "content": [{"text": f"CALL: read_log_file() 返回：\n{log_content}"}]})
        conversation_history.append({"role": "user", "content": [{"text": f"分析日志\n{log_content}"}]})
        result = call_qwen_vl(conversation_history)
        conversation_history.append({"role": "assistant", "content": [{"text": result}]})
        return {"text": result, "images": []}, conversation_history

    if "CALL: analyze_images_by_species(" in result:
        match = re.search(r'CALL: analyze_images_by_species\("(.+?)"\)', result)
        if match:
            species = match.group(1)
            analysis_result = analyze_images_by_species(species)
            conversation_history.append({"role": "assistant", "content": [{"text": analysis_result["text"]}]})
            return analysis_result, conversation_history

    # 如果没有函数调用，直接返回默认回复
    if not result.strip():
        result = "你好！我是你的野生动物保护助手。请告诉我如何帮助你（例如'现在几点'或'分析 Dog 的图像'）。"
    conversation_history.append({"role": "assistant", "content": [{"text": result}]})
    return {"text": result, "images": []}, conversation_history

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 处理用户上传的图像并分析
@app.route('/analyze', methods=['POST'])
def analyze():
    logger.debug("接收到 /analyze 请求")
    if 'image' not in request.files:
        logger.error("未检测到上传的图像")
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        logger.error("图像文件名为空")
        return jsonify({'error': 'No image selected'}), 400

    try:
        # 将图像转换为 base64 编码
        img = Image.open(image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 调用 API 分析图像
        image_url = f"data:image/png;base64,{image_base64}"
        result = analyze_image(image_url)

        # 检测异常并通知管理员
        if "异常检测：是" in result:
            notify_admin(f"检测到异常行为：{result}")

        # 初始化对话历史
        conversation_history = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": "你是野生动物保护的智能助手，任务是分析输入的图像，识别动物种类、行为模式，并检测是否存在异常行为（如入侵者、受伤动物或异常活动）。请按以下格式输出：\n- **动物种类**：识别出的动物。\n- **行为模式**：描述动物的正常或异常行为。\n- **异常检测**：是/否，说明异常原因（如有）。\n- **建议**：如检测到异常，提出通知管理员的具体建议。\n\n输入图像：{image_url}"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"text": result}]
            }
        ]

        return jsonify({
            'response': result,
            'image_base64': image_base64,
            'conversation_history': conversation_history
        })
    except Exception as e:
        logger.error(f"图像处理或分析失败：{e}")
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

# 处理用户文字输入（实时对话）
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    conversation_history = data.get('conversation_history', [])

    logger.debug(f"Received data: {data}")

    if not user_input:
        logger.error("缺少用户输入")
        return jsonify({'error': 'Missing message'}), 400

    try:
        result, updated_history = continue_conversation(user_input, conversation_history)
        logger.debug(f"返回给前端的结果：{result}")
        return jsonify({
            'response': result['text'],
            'images': result['images'],
            'conversation_history': updated_history
        })
    except Exception as e:
        logger.error(f"对话处理失败：{e}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
