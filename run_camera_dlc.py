import argparse
from pathlib import Path
import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from flask import Flask, Response, render_template_string, request
from api_infer import SnpeContext, Runtime, PerfProfile, LogLevel  # SNPE推理模块
from config import CLASSES_DET, COLORS
from utils import letterbox, det_postprocess_nms
from track import CustomTracker

app = Flask(__name__)

# 原始参数（1080*1920）
CAMERA_MATRIX = np.array([
    [1.54260292e+03, 0.00000000e+00, 9.79284155e+02],
    [0.00000000e+00, 1.54339791e+03, 6.57131835e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

DIST_COEFFS = np.array([-0.37060393, 0.04148584, -0.00094008, -0.00232051, 0.05975394])
# cv2相机默认图像尺寸480*640
scale_x = 640 / 1920
scale_y = 480 / 1080

# 适配（480*640)
NEW_CAMERA_MATRIX = CAMERA_MATRIX.copy()
NEW_CAMERA_MATRIX[0, 0] *= scale_x  # fx
NEW_CAMERA_MATRIX[1, 1] *= scale_y  # fy
NEW_CAMERA_MATRIX[0, 2] *= scale_x  # cx
NEW_CAMERA_MATRIX[1, 2] *= scale_y  # cy

# 全局变量控制畸变校正状态
use_undistort = False


def generate_frames(args):
    """生成视频帧并进行目标检测与跟踪，支持畸变校正切换"""
    global use_undistort
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
    fps_update_interval = 1.0  # 每秒更新一次FPS

    # RTSP流配置
    if args.rtsp_url:
        # 设置RTSP传输参数
        cap = cv2.VideoCapture(args.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # 增加缓冲区大小
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        cap.set(cv2.CAP_PROP_FPS, target_fps)
    else:
        cap = cv2.VideoCapture(args.camera_id)

    # 设置摄像头分辨率和其他参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)      # 增加缓冲区大小
    cap.set(cv2.CAP_PROP_FPS, 30)            # 设置帧率

    if not cap.isOpened():
        print(f"无法打开摄像头 {'RTSP URL: ' + args.rtsp_url if args.rtsp_url else 'ID: ' + str(args.camera_id)}")
        return

    # 创建帧缓冲队列
    frame_buffer = []
    max_buffer_size = 3
    last_valid_frame = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5

    tracker = CustomTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    try:
        while True:
            frame_start = time.time()
            
            # 自适应跳帧
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            # 读取帧
            for _ in range(5):
                ret, bgr = cap.read()
                # if ret:
                    # print(f"当前帧分辨率: {bgr.shape[1]}x{bgr.shape[0]}")  # 打印分辨率
                if not ret:
                    if args.rtsp_url:
                        print(f"无法获取帧，尝试重新连接... (尝试 {reconnect_attempts + 1}/{max_reconnect_attempts})")
                        cap.release()
                        time.sleep(0.5)  # 增加重连等待时间
                        cap = cv2.VideoCapture(args.rtsp_url)
                        # 重新设置RTSP参数
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                        cap.set(cv2.CAP_PROP_FPS, target_fps)
                        
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnect_attempts:
                            print("达到最大重连次数，退出程序")
                            break
                        continue
                    else:
                        print("无法获取帧")
                        break

            # 重置重连计数
            reconnect_attempts = 0

            # 检查帧是否有效
            if bgr is None or bgr.size == 0:
                if last_valid_frame is not None:
                    bgr = last_valid_frame.copy()
                else:
                    continue

            # 打印当前帧分辨率
            # print(f"当前帧分辨率: {bgr.shape[1]}x{bgr.shape[0]}")

            # 更新有效帧
            last_valid_frame = bgr.copy()

            # 更新FPS计算
            current_time = time.time()
            frame_times.append(current_time - frame_start)
            if current_time - last_fps_update >= fps_update_interval:
                if frame_times:
                    current_fps = len(frame_times) / sum(frame_times)
                    # 自适应调整跳帧数
                    if current_fps < target_fps * 0.8:
                        skip_frames = min(skip_frames + 1, 3)
                    elif current_fps > target_fps * 0.95:
                        skip_frames = max(skip_frames - 1, 0)
                frame_times.clear()
                last_fps_update = current_time

            # 处理帧
            draw = bgr.copy()

            # 畸变校正（按需）
            pre_start = time.time()
            undistort_time = 0
            if use_undistort:
                undistort_start = time.time()
                bgr = cv2.undistort(bgr, NEW_CAMERA_MATRIX, DIST_COEFFS)
                undistort_end = time.time()
                undistort_time = undistort_end - undistort_start
                draw = bgr.copy()

            # 预处理
            try:
                bgr, ratio, dwdh = letterbox(bgr, (args.W, args.H))
                dwdh = np.array(dwdh * 2, dtype=np.float32)
                # 直接使用BGR进行推理，避免颜色空间转换
                tensor = np.ascontiguousarray(bgr).astype(np.float32) / 255
            except Exception as e:
                print(f"预处理错误: {e}")
                if last_valid_frame is not None:
                    bgr = last_valid_frame.copy()
                    continue
            pre_end = time.time()

            # 推理
            inf_start = time.time()
            try:
                input_feed = {"images": tensor}
                output_names = []
                outputs = snpe_ort.Execute(output_names, input_feed)
            except Exception as e:
                print(f"推理错误: {e}")
                continue
            inf_end = time.time()

            # 后处理
            post_start = time.time()
            try:
                data = outputs['outputs']
                data = np.array(data)
                data = data.reshape(1, -1, 84)
                # 使用更快的NMS实现，提高置信度阈值减少误检
                bboxes, scores, labels = det_postprocess_nms(data, conf_thres=0.35, iou_thres=0.45)
            except Exception as e:
                print(f"后处理错误: {e}")
                continue
            post_end = time.time()

            # 处理检测结果
            if bboxes.size == 0:
                detections = np.empty((0, 6))
            else:
                bboxes -= dwdh
                bboxes /= ratio
                detections = np.column_stack((bboxes, scores, labels))

            # 更新跟踪器
            track_start = time.time()
            tracks = tracker.update(detections)
            track_end = time.time()

            # 绘制边界框和跟踪ID
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                score, label = track[5], int(track[6])
                cls = CLASSES_DET[label] if label < len(CLASSES_DET) else "Unknown"
                color = COLORS[cls] if cls in COLORS else (255, 255, 255)

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'ID: {track_id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(draw, f'{cls}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # 计算最近10帧的平均帧率
            frame_times.append(time.time() - frame_start)
            if len(frame_times) > 10:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            cv2.putText(draw, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # 显示时间信息
            read_time = current_time - frame_start
            pre_time = pre_end - pre_start
            inf_time = inf_end - inf_start
            post_time = post_end - post_start
            track_time = track_end - track_start
            cv2.putText(draw, f"Frame Read: {read_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Preprocess: {pre_time:.3f}s", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Inference: {inf_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Postprocess: {post_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Tracking: {track_time:.3f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Undistort: {undistort_time:.3f}s", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示畸变校正状态
            status = "Undistorted" if use_undistort else "Distorted"
            cv2.putText(draw, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 编码帧为JPEG
            ret, buffer = cv2.imencode('.jpg', draw)
            frame = buffer.tobytes()

            # 生成MJPEG流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        cap.release()
        # 计算并打印最后10帧的平均FPS
        final_avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
        print(f"最后10帧平均FPS: {final_avg_fps:.2f}")


@app.route('/control', methods=['POST'])
def control():
    """处理畸变校正控制请求"""
    global use_undistort
    action = request.form.get('action')
    print(f"Received action: {action}")  # 调试日志
    if action == 'undistort':
        use_undistort = True
    elif action == 'distort':
        use_undistort = False
    return '', 204


@app.route('/')
def index():
    """渲染HTML页面显示视频流"""
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv8 实时检测</title>
            <script>
                // 监听键盘事件以控制畸变校正
                document.addEventListener('keydown', function(event) {
                    if (event.key === 'u') {
                        fetch('/control', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: 'action=undistort'
                        }).then(response => console.log('Undistort enabled'));
                    } else if (event.key === 'd') {
                        fetch('/control', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: 'action=distort'
                        }).then(response => console.log('Distort enabled'));
                    } else if (event.key === 'q') {
                        alert('退出功能暂不支持，请关闭浏览器窗口或停止服务器。');
                    }
                });
            </script>
        </head>
        <body>
            <h1>YOLOv8 实时检测与跟踪</h1>
            <img src="{{ url_for('video_feed') }}">
            <p>按 'u' 启用畸变校正，按 'd' 禁用畸变校正，按 'q' 退出</p>
            <!-- 添加按钮控制（可选） -->
            <button onclick="sendAction('undistort')">启用畸变校正</button>
            <button onclick="sendAction('distort')">禁用畸变校正</button>
            <script>
                function sendAction(action) {
                    fetch('/control', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: `action=${action}`
                    }).then(response => console.log(`${action} sent`));
                }
            </script>
        </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    """提供视频流"""
    args = parse_args()
    return Response(generate_frames(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 实时目标检测与跟踪")
    parser.add_argument('--dlc-model', type=str, required=True, help='DLC模型文件路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--rtsp-url', type=str, default='', help='RTSP摄像头URL')
    parser.add_argument('--runtime', type=str, choices=['cpu', 'gpu', 'dsp'], default='cpu',
                        help='运行时后端 (cpu, gpu, 或 dsp)')
    parser.add_argument('--H', type=int, default=480, help='推理尺寸H')
    parser.add_argument('--W', type=int, default=640, help='推理尺寸W')
    parser.add_argument('--skip-frames', type=int, default=0, help='跳帧数，用于降低延迟')
    parser.add_argument('--buffer-size', type=int, default=1, help='RTSP缓冲区大小')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run(host='0.0.0.0', port=5001, threaded=True)


# python3 run_camera_dlc.py --dlc-model yolov8_int8_480_640_Q.dlc --camera-id 2 --runtime dsp --H 480 --W 640
# python3 run_camera_dlc.py --dlc-model yolov8_int8_320_320_Q.dlc --camera-id 2 --runtime dsp --H 320 --W 320


