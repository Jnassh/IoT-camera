import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/fibo/IoT')
sys.path.append('/home/fibo/IoT/IoT_device')  # 添加IoT_device目录到Python路径

import cv2
import numpy as np
import time
import datetime
import threading
import logging
import subprocess
import serial
import re
import argparse

# 华为云IoT/OBS相关
sys.path.append('/home/fibo/IoT/huaweicloud-sdk-python-obs-master/src')
from obs import ObsClient, PutObjectHeader
from client.IoT_client_config import IoTClientConfig
from client.IoT_client import IotClient
from request.services_properties import ServicesProperties
from data_logger import DataLogger

# run_camera_dlc相关
from utils import letterbox, det_postprocess_nms
from config import CLASSES_DET, COLORS
from track import CustomTracker
from api_infer import SnpeContext, Runtime, PerfProfile, LogLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量用于温湿度显示
temperature_display = None
humidity_display = None

# 全局变量控制畸变校正状态
use_undistort = False

# 相机标定参数（从run_camera_dlc.py移植）
# 原始参数（1080*1920）
CAMERA_MATRIX = np.array([
    [1.54260292e+03, 0.00000000e+00, 9.79284155e+02],
    [0.00000000e+00, 1.54339791e+03, 6.57131835e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

DIST_COEFFS = np.array([-0.37060393, 0.04148584, -0.00094008, -0.00232051, 0.05975394])

def get_camera_matrix_for_resolution(width, height):
    """根据实际分辨率动态调整相机矩阵"""
    # 原始标定分辨率
    original_width = 1920
    original_height = 1080
    
    # 计算缩放比例
    scale_x = width / original_width
    scale_y = height / original_height
    
    # 创建新的相机矩阵
    new_camera_matrix = CAMERA_MATRIX.copy()
    new_camera_matrix[0, 0] *= scale_x  # fx
    new_camera_matrix[1, 1] *= scale_y  # fy
    new_camera_matrix[0, 2] *= scale_x  # cx
    new_camera_matrix[1, 2] *= scale_y  # cy
    
    return new_camera_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlc-model', type=str, required=True, help='DLC模型路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头设备ID')
    parser.add_argument('--runtime', type=str, default='dsp', help='运行设备')
    parser.add_argument('--H', type=int, default=320, help='输入高度')
    parser.add_argument('--W', type=int, default=320, help='输入宽度')
    parser.add_argument('--buffer-size', type=int, default=3, help='缓冲区大小')
    parser.add_argument('--skip-frames', type=int, default=4, help='跳帧数')
    return parser.parse_args()

def init_serial():
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
    try:
        if ser.in_waiting:
            data = ser.readline().decode('utf-8').strip()
            print(f"串口原始数据: {data}")
            match = re.search(r'Temperature:\s*(\d+)C,\s*Humidity:\s*(\d+)%', data)
            if match:
                temperature = float(match.group(1))
                humidity = float(match.group(2))
                return humidity, temperature
            else:
                logger.error(f'数据格式错误: {data}')
                return None, None
        return None, None
    except Exception as e:
        logger.error(f'读取错误: {str(e)}')
        return None, None

def report_switch_status(iot_client, status):
    service_property = ServicesProperties()
    service_property.add_service_property(service_id="State_Connect", property='Connecting', value=status)
    iot_client.report_properties(service_properties=service_property.service_property, qos=1)
    logger.info(f"已上报摄像头连接状态: {status}")

def report_temp_humid(iot_client, temperature, humidity):
    service_property = ServicesProperties()
    service_property.add_service_property(service_id="Data_Sensor", property='Temperature', value=temperature)
    service_property.add_service_property(service_id="Data_Sensor", property='Humidity', value=humidity)
    iot_client.report_properties(service_properties=service_property.service_property, qos=1)
    logger.info(f"已上报温湿度数据: 温度={temperature}°C, 湿度={humidity}%")

def temp_humid_worker(ser, iot_client, stop_event, data_logger):
    global temperature_display, humidity_display
    while not stop_event.is_set():
        humidity, temperature = read_dht11(ser)
        if humidity is not None and temperature is not None:
            temperature_display = temperature
            humidity_display = humidity
            report_temp_humid(iot_client, temperature, humidity)
            data_logger.log_data(temperature, humidity)
            data_logger._upload_to_cloud()
        stop_event.wait(2)

def upload_to_obs(local_file_path, object_key):
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

def camera_worker(device_id, dlc_model, iot_client, data_logger, args):
    global temperature_display, humidity_display, use_undistort
    # SNPE初始化
    runtime = Runtime.DSP if args.runtime.lower() == 'dsp' else Runtime.CPU
    snpe_ort = SnpeContext(dlc_model, [], runtime, PerfProfile.HIGH_PERFORMANCE, LogLevel.ERROR)
    assert snpe_ort.Initialize() == 0, "SNPE初始化失败"
    H, W = args.H, args.W
    
    # 初始化USB摄像头
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头设备 {device_id}")
        return
        
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffer_size)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 验证摄像头参数是否设置成功
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"摄像头参数: 分辨率={actual_width}x{actual_height}, FPS={actual_fps}")
    
    # 根据实际分辨率获取相机矩阵
    camera_matrix = get_camera_matrix_for_resolution(actual_width, actual_height)
    logger.info(f"使用的相机矩阵: {camera_matrix}")
    
    tracker = CustomTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    connected = False

    # 创建可调整大小的窗口
    cv2.namedWindow('USB Camera', cv2.WINDOW_NORMAL)
    # 设置初始窗口大小
    cv2.resizeWindow('USB Camera', 1280, 720)

    # FPS计算相关变量
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5  # 每0.5秒更新一次FPS

    # 检查连接
    start_time = time.time()
    while not cap.isOpened():
        if time.time() - start_time > 10:
            print("摄像头连接超时")
            report_switch_status(iot_client, 0)
            return
        time.sleep(0.1)
    print("摄像头连接成功")
    report_switch_status(iot_client, 1)
    connected = True

    try:
        while True:
            # 跳帧处理
            for _ in range(args.skip_frames):
                cap.grab()
                
            ret, bgr = cap.read()
            if not ret or bgr is None:
                print("无法获取画面")
                if connected:
                    report_switch_status(iot_client, 0)
                    connected = False
                break

            # 计算FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= fps_update_interval:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time

            # 畸变校正（按需）- 在复制draw之前进行
            undistort_time = 0
            if use_undistort:
                undistort_start = time.time()
                try:
                    # 使用动态调整的相机矩阵
                    bgr = cv2.undistort(bgr, camera_matrix, DIST_COEFFS)
                    undistort_end = time.time()
                    undistort_time = undistort_end - undistort_start
                    logger.debug(f"畸变校正完成，耗时: {undistort_time:.3f}s")
                except Exception as e:
                    logger.error(f"畸变校正失败: {e}")
                    use_undistort = False

            # 复制矫正后的图像用于显示
            draw = bgr.copy()

            # 预处理
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            dwdh = np.array(dwdh * 2, dtype=np.float32)
            tensor = np.ascontiguousarray(bgr).astype(np.float32) / 255

            # 推理
            input_feed = {"images": tensor}
            output_names = []
            outputs = snpe_ort.Execute(output_names, input_feed)

            # 后处理
            data = outputs['outputs']
            data = np.array(data)
            data = data.reshape(1, -1, 84)
            bboxes, scores, labels = det_postprocess_nms(data, conf_thres=0.35, iou_thres=0.45)

            if bboxes.size == 0:
                detections = np.empty((0, 6))
            else:
                bboxes -= dwdh
                bboxes /= ratio
                detections = np.column_stack((bboxes, scores, labels))
            tracks = tracker.update(detections)

            # 绘制检测框
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                score, label = track[5], int(track[6])
                cls = CLASSES_DET[label] if label < len(CLASSES_DET) else "Unknown"
                color = COLORS[cls] if cls in COLORS else (255, 255, 255)
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(draw, f'{cls}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # 显示温湿度和FPS信息
            temp_str = f"Temp:{temperature_display if temperature_display is not None else '--'}C"
            humid_str = f"Humid:{humidity_display if humidity_display is not None else '--'}%"
            fps_text = f"FPS: {fps:.1f}"
            text = f"{temp_str} {humid_str} | {fps_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5  # 减小字体大小
            thickness = 1     # 减小字体粗细
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            h, w = draw.shape[:2]
            x = 10           # 左边距10像素
            y = h - 10      # 底部边距10像素
            cv2.putText(draw, text, (x, y), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)

            # 显示畸变校正状态和时间信息
            status = "Undistorted" if use_undistort else "Distorted"
            cv2.putText(draw, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if use_undistort:
                cv2.putText(draw, f"Undistort: {undistort_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示操作提示
            cv2.putText(draw, "Press 'u' to enable undistort, 'd' to disable", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('USB Camera', draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                try:
                    img_path = "/home/fibo/IoT/picture/upload_frame.jpg"
                    cv2.imwrite(img_path, draw)
                    object_key = f"camera_upload/{int(time.time())}.jpg"
                    resp = upload_to_obs(img_path, object_key)
                    if resp.status < 300:
                        print("图片上传成功")
                    else:
                        print("图片上传失败", resp.errorMessage)
                except Exception as e:
                    print(f"保存图片时出错: {e}")
                    continue
            elif key == ord('u'):
                use_undistort = True
                print("畸变校正已启用")
            elif key == ord('d'):
                use_undistort = False
                print("畸变校正已禁用")

    except KeyboardInterrupt:
        print('\n程序已退出')
    finally:
        report_switch_status(iot_client, 0)
        cap.release()
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    ser = init_serial()
    if not ser:
        logger.error('无法初始化串口，程序退出')
        return
    data_logger = DataLogger()
    
    # IoT客户端配置和连接
    try:
        logger.info("正在初始化IoT客户端...")
        client_cfg = IoTClientConfig(
            server_ip='3c62575c78.st1.iotda-device.cn-east-3.myhuaweicloud.com',
            device_id='6837094c94a9a05c3361f6c7_Camera_001',
            secret='12345678',
            is_ssl=False
        )
        iot_client = IotClient(client_cfg)
        
        logger.info("正在连接IoT平台...")
        iot_client.connect()
        
        # 等待连接成功
        time.sleep(2)  # 给连接一些时间
        
        logger.info("正在启动IoT客户端...")
        iot_client.start()
        
        # 等待客户端启动
        time.sleep(1)
        
        logger.info("IoT平台连接成功")
    except Exception as e:
        logger.error(f"IoT平台连接出错: {str(e)}")
        return

    stop_event = threading.Event()
    th_thread = threading.Thread(target=temp_humid_worker, args=(ser, iot_client, stop_event, data_logger))
    th_thread.start()
    logger.info("温湿度监测线程已启动")

    # 使用USB摄像头
    camera_worker(args.camera_id, args.dlc_model, iot_client, data_logger, args)

    stop_event.set()
    th_thread.join()
    if ser:
        ser.close()
    # 程序退出前确保数据上传
    data_logger._upload_to_cloud()
    logger.info("程序退出，数据已保存并上传")

if __name__ == "__main__":
    main()
