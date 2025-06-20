import argparse
from pathlib import Path
import cv2
import numpy as np
import time
import onnxruntime as ort
from flask import Flask, Response, render_template_string
from track import CustomTracker

from config import CLASSES_DET, COLORS
from utils import blob, letterbox, det_postprocess_nms

app = Flask(__name__)



def generate_frames(args):
    """生成视频帧并进行目标检测与跟踪"""
    # ONNX初始化
    session = ort.InferenceSession(args.onnx_model)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # 推理尺寸
    H, W = args.H, args.W

    # 打开摄像头
    if args.rtsp_url:
        cap = cv2.VideoCapture(args.rtsp_url)
    else:
        cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {'RTSP URL: ' + args.rtsp_url if args.rtsp_url else 'ID: ' + str(args.camera_id)}")
        return

    # 设置RTSP缓冲区大小
    if args.rtsp_url:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # 用于计算最近10帧的平均帧率
    frame_times = []
    tracker = CustomTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    try:
        while True:
            frame_start = time.time()  # 记录每帧开始时间
            read_start = time.time()
            ret, bgr = cap.read()
            read_end = time.time()
            if not ret:
                if args.rtsp_url:
                    print("无法获取帧，尝试重新连接...")
                    cap.release()
                    cap = cv2.VideoCapture(args.rtsp_url)
                    time.sleep(1)  # 等待1秒后重试
                    continue
                else:
                    print("无法获取帧")
                    break

            draw = bgr.copy()

            # 预处理
            pre_start = time.time()
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            dwdh = np.array(dwdh * 2, dtype=np.float32)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(rgb).astype(np.float32) / 255
            pre_end = time.time()

            # 推理
            inf_start = time.time()
            outputs = session.run(output_names, {input_name: tensor})
            inf_end = time.time()

            # 后处理
            post_start = time.time()
            data = outputs[0]
            data = data.reshape(1, -1, 84)
            bboxes, scores, labels = det_postprocess_nms(data)
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
            read_time = read_end - read_start
            pre_time = pre_end - pre_start
            inf_time = inf_end - inf_start
            post_time = post_end - post_start
            track_time = track_end - track_start
            
            cv2.putText(draw, f"Preprocess: {pre_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(draw, f"Inference: {inf_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if args.rtsp_url:
                cv2.putText(draw, f"RTSP URL: {args.rtsp_url}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv8 Detection</title>
        </head>
        <body>
            <h1>YOLOv8 Real-time Detection</h1>
            <img src="{{ url_for('video_feed') }}" width="640">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    args = parse_args()
    return Response(generate_frames(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX实时目标检测与跟踪")
    parser.add_argument('--onnx-model', type=str, required=True, help='ONNX模型文件路径')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--rtsp-url', type=str, default='', help='RTSP摄像头URL')
    parser.add_argument('--H', type=int, default=480, help='推理尺寸H')
    parser.add_argument('--W', type=int, default=640, help='推理尺寸W')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    app.run(host='0.0.0.0', port=5001, threaded=True)

# python3 run_camera_onnx.py --onnx-model yolov8n.onnx --camera-id 2
