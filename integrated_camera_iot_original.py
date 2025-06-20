import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/fibo/IoT')
sys.path.append('/home/fibo/IoT/IoT_device')

import argparse
from pathlib import Path
import cv2
import numpy as np
import time
import datetime
import threading
import logging
import serial
import re
from flask import Flask, Response, render_template_string, request, jsonify
import dashscope
import base64
from io import BytesIO
from datetime import datetime, timedelta
import signal
import csv

# 华为云IoT/OBS相关导入
sys.path.append('/home/fibo/IoT/huaweicloud-sdk-python-obs-master/src')
from obs import ObsClient, PutObjectHeader
from client.IoT_client_config import IoTClientConfig
from client.IoT_client import IotClient
from request.services_properties import ServicesProperties
from data_logger import DataLogger

# YOLO检测相关导入
from api_infer import SnpeContext, Runtime, PerfProfile, LogLevel
from config import CLASSES_DET, COLORS
from utils import letterbox, det_postprocess_nms
from track import CustomTracker

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量
temperature_display = None
humidity_display = None
use_undistort = False
last_frame = None
iot_client = None
data_logger = None
conversation_history = []
sensor_logger = None
log_file = "log.txt"

# 配置通义千问API
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    logger.warning("API Key 未提供，使用默认密钥")
    dashscope.api_key = "sk-cc0d38bc65a447aaa86c2717a2106d37"
else:
    dashscope.api_key = api_key
    logger.debug("成功加载 API Key")

# 相机标定参数
CAMERA_MATRIX = np.array([
    [1.54260292e+03, 0.00000000e+00, 9.79284155e+02],
    [0.00000000e+00, 1.54339791e+03, 6.57131835e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

DIST_COEFFS = np.array([-0.37060393, 0.04148584, -0.00094008, -0.00232051, 0.05975394])

def get_camera_matrix_for_resolution(width, height):
    """根据实际分辨率动态调整相机矩阵"""
    original_width = 1920
    original_height = 1080
    scale_x = width / original_width
    scale_y = height / original_height
    new_camera_matrix = CAMERA_MATRIX.copy()
    new_camera_matrix[0, 0] *= scale_x
    new_camera_matrix[1, 1] *= scale_y
    new_camera_matrix[0, 2] *= scale_x
    new_camera_matrix[1, 2] *= scale_y
    return new_camera_matrix

def init_serial():
    """初始化串口连接"""
    try:
        ser = serial.Serial(
            port='/dev/ttyHS1',
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        return ser
    except Exception as e:
        logger.error(f'串口初始化错误: {str(e)}')
        return None

def read_dht11(ser):
    """读取DHT11温湿度数据"""
    try:
        if ser.in_waiting:
            data = ser.readline().decode('utf-8').strip()
            match = re.search(r'Temperature:\s*(\d+)C,\s*Humidity:\s*(\d+)%', data)
            if match:
                temperature = float(match.group(1))
                humidity = float(match.group(2))
                return humidity, temperature
        return None, None
    except Exception as e:
        logger.error(f'读取错误: {str(e)}')
        return None, None

def report_switch_status(status):
    """向IoT平台报告摄像头连接状态"""
    if iot_client:
        service_property = ServicesProperties()
        service_property.add_service_property(service_id="State_Connect", property='Connecting', value=status)
        iot_client.report_properties(service_properties=service_property.service_property, qos=1)
        logger.info(f"已上报摄像头连接状态: {status}")

def report_temp_humid(temperature, humidity):
    """向IoT平台报告温湿度数据"""
    if iot_client:
        service_property = ServicesProperties()
        service_property.add_service_property(service_id="Data_Sensor", property='Temperature', value=temperature)
        service_property.add_service_property(service_id="Data_Sensor", property='Humidity', value=humidity)
        iot_client.report_properties(service_properties=service_property.service_property, qos=1)
        logger.info(f"已上报温湿度数据: 温度={temperature}°C, 湿度={humidity}%")

def temp_humid_worker(ser, stop_event):
    """温湿度数据采集和上报线程"""
    global temperature_display, humidity_display
    last_record_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time()
        humidity, temperature = read_dht11(ser)
        
        if humidity is not None and temperature is not None:
            temperature_display = temperature
            humidity_display = humidity
            
            # 向IoT平台报告数据
            report_temp_humid(temperature, humidity)
            
            # 每5秒记录一次数据
            if current_time - last_record_time >= 5:
                if sensor_logger:
                    sensor_logger.log_data(temperature, humidity)
                last_record_time = current_time
        
        # 等待时间缩短，以便更及时地响应停止信号
        stop_event.wait(1)

def upload_to_obs(local_file_path, object_key):
    """上传图片到华为云OBS"""
    try:
        ak = "HPUARTT7GSHGOO7FDKNK"
        sk = "wLaxa0pyapOfwf7tuL0hgI868EMMPVqUr4TOI8zr"
        server = "https://obs.cn-south-1.myhuaweicloud.com"
        bucketName = "storage-videophoto"
        obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
        headers = PutObjectHeader()
        headers.contentType = 'image/jpeg'
        metadata = {'meta1': 'value1', 'meta2': 'value2'}
        resp = obsClient.putFile(bucketName, object_key, local_file_path, metadata, headers)
        return resp
    except Exception as e:
        logger.error(f"上传到OBS失败: {e}")
        return None

def generate_frames(args):
    """生成视频帧并进行目标检测与跟踪"""
    global use_undistort, last_frame, temperature_display, humidity_display

    # SNPE初始化
    runtime_map = {'cpu': Runtime.CPU, 'gpu': Runtime.GPU, 'dsp': Runtime.DSP}
    selected_runtime = runtime_map[args.runtime.lower()]
    snpe_ort = SnpeContext(args.dlc_model, [], selected_runtime, PerfProfile.HIGH_PERFORMANCE, LogLevel.ERROR)
    assert snpe_ort.Initialize() == 0, "SNPE初始化失败"

    # 推理尺寸
    H, W = args.H, args.W

    # 性能监控变量
    frame_count = 0
    skip_frames = args.skip_frames
    target_fps = 30
    frame_times = []
    last_fps_update = time.time()
    fps_update_interval = 1.0

    # 摄像头初始化
    if args.rtsp_url:
        cap = cv2.VideoCapture(args.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
    else:
        cap = cv2.VideoCapture(args.camera_id)

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        logger.error(f"无法打开摄像头")
        report_switch_status(0)
        return

    report_switch_status(1)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_matrix = get_camera_matrix_for_resolution(actual_width, actual_height)

    tracker = CustomTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    last_valid_frame = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5

    try:
        while True:
            frame_start = time.time()
            
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            ret, bgr = cap.read()
            if not ret:
                if args.rtsp_url:
                    logger.warning(f"无法获取帧，尝试重新连接... (尝试 {reconnect_attempts + 1}/{max_reconnect_attempts})")
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(args.rtsp_url)
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        logger.error("达到最大重连次数，退出程序")
                        break
                    continue
                else:
                    logger.error("无法获取帧")
                    break

            reconnect_attempts = 0

            if bgr is None or bgr.size == 0:
                if last_valid_frame is not None:
                    bgr = last_valid_frame.copy()
                else:
                    continue

            last_valid_frame = bgr.copy()
            draw = bgr.copy()

            # 畸变校正
            undistort_time = 0
            if use_undistort:
                undistort_start = time.time()
                bgr = cv2.undistort(bgr, camera_matrix, DIST_COEFFS)
                undistort_end = time.time()
                undistort_time = undistort_end - undistort_start
                draw = bgr.copy()

            # 预处理
            try:
                bgr, ratio, dwdh = letterbox(bgr, (W, H))
                dwdh = np.array(dwdh * 2, dtype=np.float32)
                tensor = np.ascontiguousarray(bgr).astype(np.float32) / 255
            except Exception as e:
                logger.error(f"预处理错误: {e}")
                continue

            # 推理
            try:
                input_feed = {"images": tensor}
                output_names = []
                outputs = snpe_ort.Execute(output_names, input_feed)
            except Exception as e:
                logger.error(f"推理错误: {e}")
                continue

            # 后处理
            try:
                data = outputs['outputs']
                data = np.array(data)
                data = data.reshape(1, -1, 84)
                bboxes, scores, labels = det_postprocess_nms(data, conf_thres=0.35, iou_thres=0.45)
            except Exception as e:
                logger.error(f"后处理错误: {e}")
                continue

            if bboxes.size == 0:
                detections = np.empty((0, 6))
            else:
                bboxes -= dwdh
                bboxes /= ratio
                detections = np.column_stack((bboxes, scores, labels))

            tracks = tracker.update(detections)

            # 绘制检测框和跟踪ID
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                score, label = track[5], int(track[6])
                cls = CLASSES_DET[label] if label < len(CLASSES_DET) else "Unknown"
                color = COLORS[cls] if cls in COLORS else (255, 255, 255)
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(draw, f'{cls}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # 显示温湿度信息
            temp_str = f"Temp:{temperature_display if temperature_display is not None else '--'}C"
            humid_str = f"Humid:{humidity_display if humidity_display is not None else '--'}%"
            cv2.putText(draw, f"{temp_str} {humid_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示畸变校正状态
            status = "Undistorted" if use_undistort else "Distorted"
            cv2.putText(draw, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 更新FPS
            current_time = time.time()
            frame_times.append(current_time - frame_start)
            if current_time - last_fps_update >= fps_update_interval:
                if frame_times:
                    fps = len(frame_times) / sum(frame_times)
                    cv2.putText(draw, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frame_times.clear()
                last_fps_update = current_time

            last_frame = draw.copy()

            # 编码帧为JPEG
            ret, buffer = cv2.imencode('.jpg', draw)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        logger.error(f"视频流处理错误: {e}")
    finally:
        cap.release()
        report_switch_status(0)

@app.route('/')
def index():
    """渲染Web界面"""
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>集成监控系统</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .controls { margin: 20px 0; }
                button { margin: 5px; padding: 10px; }
                .video-container { margin-top: 20px; }
                .analysis-result { 
                    margin-top: 20px; 
                    padding: 10px; 
                    background-color: #f0f0f0; 
                    border-radius: 5px;
                }
                .chat-container {
                    margin-top: 20px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                #chatInput {
                    width: 80%;
                    padding: 5px;
                    margin-right: 10px;
                }
            </style>
            <script>
                function sendAction(action) {
                    fetch('/control', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: `action=${action}`
                    }).then(response => console.log(`${action} sent`));
                }

                function captureImage() {
                    fetch('/capture', {
                        method: 'POST'
                    }).then(response => {
                        if(response.ok) {
                            alert('图片已保存并上传到云端');
                        } else {
                            alert('保存图片失败');
                        }
                    });
                }

                function analyzeCurrentFrame() {
                    fetch('/analyze_frame', {
                        method: 'POST'
                    }).then(response => response.json())
                    .then(data => {
                        if(data.error) {
                            alert('分析失败：' + data.error);
                        } else {
                            document.getElementById('analysisResult').innerText = data.result;
                        }
                    });
                }

                function sendMessage() {
                    const input = document.getElementById('chatInput');
                    const message = input.value;
                    if (!message) return;

                    fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            conversation_history: window.conversationHistory || []
                        })
                    }).then(response => response.json())
                    .then(data => {
                        const chatHistory = document.getElementById('chatHistory');
                        chatHistory.innerHTML += `<p><strong>你:</strong> ${message}</p>`;
                        chatHistory.innerHTML += `<p><strong>助手:</strong> ${data.response}</p>`;
                        window.conversationHistory = data.conversation_history;
                        input.value = '';
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    });
                }

                document.addEventListener('keydown', function(event) {
                    if (event.key === 'u') {
                        sendAction('undistort');
                    } else if (event.key === 'd') {
                        sendAction('distort');
                    } else if (event.key === 's') {
                        captureImage();
                    } else if (event.key === 'a') {
                        analyzeCurrentFrame();
                    }
                });
            </script>
        </head>
        <body>
            <div class="container">
                <h1>集成监控系统</h1>
                <div class="controls">
                    <button onclick="sendAction('undistort')">启用畸变校正</button>
                    <button onclick="sendAction('distort')">禁用畸变校正</button>
                    <button onclick="captureImage()">保存图片</button>
                    <button onclick="analyzeCurrentFrame()">分析当前帧</button>
                </div>
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" style="max-width: 100%;">
                </div>
                <div class="analysis-result">
                    <h3>分析结果</h3>
                    <p id="analysisResult">点击"分析当前帧"按钮进行分析</p>
                </div>
                <div class="chat-container">
                    <h3>对话系统</h3>
                    <div id="chatHistory" style="height: 200px; overflow-y: auto; margin-bottom: 10px;"></div>
                    <div style="display: flex;">
                        <input type="text" id="chatInput" placeholder="输入消息...">
                        <button onclick="sendMessage()">发送</button>
                    </div>
                </div>
                <div class="instructions">
                    <p>快捷键说明:</p>
                    <ul>
                        <li>'u': 启用畸变校正</li>
                        <li>'d': 禁用畸变校正</li>
                        <li>'s': 保存并上传图片</li>
                        <li>'a': 分析当前帧</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    """提供视频流"""
    args = parse_args()
    return Response(generate_frames(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    """处理畸变校正控制请求"""
    global use_undistort
    action = request.form.get('action')
    if action == 'undistort':
        use_undistort = True
    elif action == 'distort':
        use_undistort = False
    return '', 204

@app.route('/capture', methods=['POST'])
def capture():
    """处理图片保存请求"""
    global last_frame
    if last_frame is not None:
        try:
            img_path = "/home/fibo/IoT/picture/upload_frame.jpg"
            cv2.imwrite(img_path, last_frame)
            object_key = f"camera_upload/{int(time.time())}.jpg"
            resp = upload_to_obs(img_path, object_key)
            if resp and resp.status < 300:
                logger.info("图片上传成功")
                return "Success", 200
            else:
                logger.error("图片上传失败")
                return "Failed", 500
        except Exception as e:
            logger.error(f"保存图片时出错: {e}")
            return "Error", 500
    return "No frame available", 400

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """分析当前帧"""
    global last_frame
    if last_frame is not None:
        try:
            result = analyze_image(last_frame)
            # 检测异常并通知管理员
            if "异常" in result.lower():
                notify_admin(f"检测到异常：{result}")
            return jsonify({'result': result})
        except Exception as e:
            logger.error(f"分析当前帧失败：{e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': '没有可用的帧'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """处理用户对话"""
    global conversation_history
    try:
        data = request.json
        user_input = data.get('message')
        conversation_history = data.get('conversation_history', [])

        if not user_input:
            return jsonify({'error': '消息为空'}), 400

        # 添加用户消息到对话历史
        conversation_history.append({
            "role": "user",
            "content": [{"text": user_input}]
        })

        # 调用API获取响应
        response = call_qwen_vl(conversation_history)

        # 添加助手响应到对话历史
        conversation_history.append({
            "role": "assistant",
            "content": [{"text": response}]
        })

        return jsonify({
            'response': response,
            'conversation_history': conversation_history
        })
    except Exception as e:
        logger.error(f"处理对话失败：{e}")
        return jsonify({'error': str(e)}), 500

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="集成监控系统")
    parser.add_argument('--dlc-model', type=str, required=True, help='DLC模型文件路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--rtsp-url', type=str, default='', help='RTSP摄像头URL')
    parser.add_argument('--runtime', type=str, choices=['cpu', 'gpu', 'dsp'], default='cpu',
                        help='运行时后端 (cpu, gpu, 或 dsp)')
    parser.add_argument('--H', type=int, default=480, help='推理尺寸H')
    parser.add_argument('--W', type=int, default=640, help='推理尺寸W')
    parser.add_argument('--skip-frames', type=int, default=0, help='跳帧数')
    parser.add_argument('--buffer-size', type=int, default=1, help='RTSP缓冲区大小')
    return parser.parse_args()

def signal_handler(signum, frame):
    """处理退出信号"""
    logger.info("接收到退出信号，正在清理资源...")
    if iot_client:
        report_switch_status(0)  # 更新摄像头状态为断开
        time.sleep(1)  # 等待状态更新
    sys.exit(0)

def main():
    """主函数"""
    global iot_client, data_logger, sensor_logger
    args = parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 初始化串口
    ser = init_serial()
    if not ser:
        logger.error('无法初始化串口')
        return

    # 初始化数据记录器
    data_logger = DataLogger()
    sensor_logger = SensorDataLogger()

    # 初始化IoT客户端
    try:
        client_cfg = IoTClientConfig(
            server_ip='3c62575c78.st1.iotda-device.cn-east-3.myhuaweicloud.com',
            device_id='6837094c94a9a05c3361f6c7_Camera_001',
            secret='12345678',
            is_ssl=False
        )
        iot_client = IotClient(client_cfg)
        iot_client.connect()
        time.sleep(2)
        iot_client.start()
        logger.info("IoT平台连接成功")
    except Exception as e:
        logger.error(f"IoT平台连接出错: {str(e)}")
        return

    # 启动温湿度监测线程
    stop_event = threading.Event()
    th_thread = threading.Thread(target=temp_humid_worker, args=(ser, stop_event))
    th_thread.start()
    logger.info("温湿度监测线程已启动")

    try:
        # 启动Web服务
        app.run(host='0.0.0.0', port=5001, threaded=True)
    except KeyboardInterrupt:
        logger.info("程序正在退出...")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
    finally:
        # 清理资源
        logger.info("正在清理资源...")
        stop_event.set()
        th_thread.join()
        
        if ser:
            ser.close()
            
        # 上传传感器数据文件
        if sensor_logger:
            logger.info("正在上传传感器数据文件...")
            sensor_logger.upload_to_obs()
            
        if iot_client:
            report_switch_status(0)  # 更新摄像头状态为断开
            time.sleep(1)  # 等待状态更新完成
            iot_client.stop()  # 停止IoT客户端
            iot_client.disconnect()  # 断开IoT连接
            
        logger.info("程序已退出，所有资源已清理")

def call_qwen_vl(messages):
    """调用通义千问API进行图像分析"""
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

def analyze_image(image):
    """分析图像中的目标"""
    try:
        # 将图像转换为base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode()
        image_url = f"data:image/jpeg;base64,{image_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": "你是一个目标检测和分析专家。请分析图像中的目标，按以下格式输出：\n1. 目标类型：检测到的目标类型\n2. 行为模式：描述目标的行为\n3. 异常检测：是否存在异常情况\n4. 建议：如果检测到异常，提供处理建议"}
                ]
            }
        ]
        return call_qwen_vl(messages)
    except Exception as e:
        logger.error(f"图像分析失败：{e}")
        return f"分析失败：{str(e)}"

def notify_admin(message):
    """通知管理员异常情况"""
    logger.info(f"通知管理员：{message}")
    # 这里可以添加实际的通知逻辑，比如发送邮件或短信
    # TODO: 实现实际的通知机制

def get_current_time():
    """获取当前时间"""
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"当前时间：{now}")
        return now
    except Exception as e:
        logger.error(f"获取时间失败：{e}")
        return f"错误：无法获取当前时间，原因：{str(e)}"

class SensorDataLogger:
    """传感器数据记录器"""
    def __init__(self):
        self.log_dir = "sensor_logs"
        self.filename = None
        self.init_log_file()
        
    def init_log_file(self):
        """初始化日志文件"""
        try:
            # 创建日志目录
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            # 使用人类可读的时间格式创建文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.filename = os.path.join(self.log_dir, f"sensor_data_{timestamp}.csv")
            
            # 创建CSV文件并写入表头
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Temperature', 'Humidity'])
            
            logger.info(f"创建传感器数据日志文件: {self.filename}")
        except Exception as e:
            logger.error(f"创建日志文件失败: {e}")
            self.filename = None

    def log_data(self, temperature, humidity):
        """记录传感器数据"""
        if not self.filename:
            return
            
        try:
            # 使用人类可读的时间格式记录数据
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_time, temperature, humidity])
        except Exception as e:
            logger.error(f"写入传感器数据失败: {e}")

    def upload_to_obs(self):
        """将日志文件上传到OBS"""
        if not self.filename or not os.path.exists(self.filename):
            logger.warning("没有可上传的传感器数据文件")
            return False

        try:
            # 上传到OBS，保持原始文件名
            object_key = f"sensor_data/{os.path.basename(self.filename)}"
            ak = "HPUARTT7GSHGOO7FDKNK"
            sk = "wLaxa0pyapOfwf7tuL0hgI868EMMPVqUr4TOI8zr"
            server = "https://obs.cn-south-1.myhuaweicloud.com"
            bucketName = "storage-videophoto"
            
            obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
            headers = PutObjectHeader()
            headers.contentType = 'text/csv'
            
            resp = obsClient.putFile(bucketName, object_key, self.filename)
            
            if resp.status < 300:
                logger.info(f"传感器数据文件上传成功: {object_key}")
                return True
            else:
                logger.error(f"传感器数据文件上传失败: {resp.errorMessage}")
                return False
        except Exception as e:
            logger.error(f"上传传感器数据文件失败: {e}")
            return False

def log_detection_event(detections, frame):
    """记录检测事件到日志文件"""
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detected_objects = []
        
        for track in detections:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            score, label = track[5], int(track[6])
            cls = CLASSES_DET[label] if label < len(CLASSES_DET) else "Unknown"
            detected_objects.append(f"{cls}(ID:{track_id})")
        
        if detected_objects:
            # 保存当前帧
            frame_filename = f"outputs/frame_{int(time.time())}.jpg"
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(frame_filename, frame)
            
            # 写入日志
            log_entry = f"{current_time}, 检测到: {', '.join(detected_objects)}, 图片: {frame_filename}"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
            
            # 分析新检测到的对象
            analysis_result = analyze_image(frame)
            if "异常" in analysis_result.lower():
                notify_admin(f"检测到异常：{analysis_result}")
    except Exception as e:
        logger.error(f"记录检测事件失败：{e}")

def read_log_file(time_range=None):
    """读取日志文件，支持时间范围过滤"""
    try:
        if not os.path.exists(log_file):
            return "日志文件不存在"
            
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        if not time_range:
            return "".join(lines)
            
        start_time, end_time = time_range
        filtered_lines = []
        for line in lines:
            try:
                timestamp_str = line.split(", ")[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                if start_time <= timestamp <= end_time:
                    filtered_lines.append(line)
            except (ValueError, IndexError):
                continue
                
        return "".join(filtered_lines) if filtered_lines else "指定时间范围内没有日志记录"
    except Exception as e:
        logger.error(f"读取日志失败：{e}")
        return f"读取日志失败：{str(e)}"

def analyze_images_by_species(species_name):
    """分析特定物种的所有图像"""
    try:
        if not os.path.exists(log_file):
            return {"text": "日志文件不存在", "images": []}
            
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        matched_images = []
        for line in lines:
            if species_name.lower() in line.lower():
                try:
                    img_path = line.strip().split(", ")[-1].split(": ")[-1]
                    if os.path.exists(img_path):
                        matched_images.append((img_path, line.strip()))
                except IndexError:
                    continue
                    
        if not matched_images:
            return {"text": f"未找到{species_name}的图像记录", "images": []}
            
        results = []
        image_base64_list = []
        for img_path, log_line in matched_images:
            try:
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                image_url = f"data:image/jpeg;base64,{img_base64}"
                analysis_result = analyze_image_with_api(image_url)
                results.append(f"日志记录：{log_line}\n分析结果：{analysis_result}\n")
                image_base64_list.append(img_base64)
            except Exception as e:
                logger.error(f"处理图像失败 {img_path}: {e}")
                
        return {
            "text": "\n".join(results) if results else f"无法分析{species_name}的图像",
            "images": image_base64_list
        }
    except Exception as e:
        logger.error(f"分析物种图像失败：{e}")
        return {"text": f"分析失败：{str(e)}", "images": []}

@app.route('/analyze_species', methods=['POST'])
def analyze_species():
    """处理特定物种分析请求"""
    try:
        data = request.json
        species_name = data.get('species')
        if not species_name:
            return jsonify({'error': '未指定物种名称'}), 400
            
        result = analyze_images_by_species(species_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"分析物种请求失败：{e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_logs', methods=['POST'])
def get_logs():
    """获取日志记录"""
    try:
        data = request.json
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        if start_time and end_time:
            try:
                start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                logs = read_log_file((start, end))
            except ValueError as e:
                return jsonify({'error': f'时间格式错误：{str(e)}'}), 400
        else:
            logs = read_log_file()
            
        return jsonify({'logs': logs})
    except Exception as e:
        logger.error(f"获取日志失败：{e}")
        return jsonify({'error': str(e)}), 500

def analyze_image_with_api(image_url):
    """使用通义千问API分析图像"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": "分析图像中的目标，按以下格式输出：\n1. 目标类型：检测到的目标类型\n2. 行为模式：描述目标的行为\n3. 异常检测：是否存在异常情况\n4. 建议：如果检测到异常，提供处理建议"}
                ]
            }
        ]
        response = dashscope.MultiModalConversation.call(
            api_key=dashscope.api_key,
            model='qwen-vl-max',
            messages=messages
        )
        
        if not hasattr(response, 'output') or not response.output:
            return "API响应缺少output"
            
        choice = response.output.choices[0]
        content = choice.message.content
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                return item['text']
        return "API未返回文本结果"
    except Exception as e:
        logger.error(f"API调用失败：{e}")
        return f"分析失败：{str(e)}"

if __name__ == '__main__':
    main()
