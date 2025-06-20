import argparse
from pathlib import Path
import cv2
import numpy as np
import time
from api_infer import *  # 导入SNPE推理相关库
from config import CLASSES_DET, COLORS
from utils import blob, letterbox, det_postprocess_nms
from track import CustomTracker

def process_image(args):
    # SNPE初始化部分
    runtime_map = {
        'cpu': Runtime.CPU,
        'gpu': Runtime.GPU,
        'dsp': Runtime.DSP
    }
    selected_runtime = runtime_map[args.runtime.lower()]
    snpe_ort = SnpeContext(args.dlc_model, [], selected_runtime, PerfProfile.BALANCED, LogLevel.INFO)
    assert snpe_ort.Initialize() == 0
    # 推理尺寸
    H, W = args.H, args.W

    start_time = time.time()

    # 读取输入图片
    img_path = args.image_path
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Cannot open image {img_path}")
        return

    # 保存原始图片的分辨率，用于最终输出
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
    input_feed = {"images": tensor}  # netron查看onnx输入名称
    output_names = []
    outputs = snpe_ort.Execute(output_names, input_feed)
    inf_end = time.time()

    # 后处理
    post_start = time.time()
    data = outputs['outputs']  # netron查看onnx输出名称
    data = np.array(data)
    data = data.reshape(1, -1, 84)  # netron查看onnx输出尺寸
    bboxes, scores, labels = det_postprocess_nms(data)
    post_end = time.time()

    # Handle detections
    if bboxes.size == 0:
        detections = np.empty((0, 6))
    else:
        bboxes -= dwdh
        bboxes /= ratio
        detections = np.column_stack((bboxes, scores, labels))

    # Drawing bounding boxes 
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        score, label = detection[4], int(detection[5])
        cls = CLASSES_DET[label] if label < len(CLASSES_DET) else "Unknown"
        color = COLORS[cls] if cls in COLORS else (255, 255, 255)

        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(draw, f'{cls}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time

    # Display timing information
    pre_time = pre_end - pre_start
    inf_time = inf_end - inf_start
    post_time = post_end - post_start

    # 在图片上显示处理时间
    cv2.putText(draw, f"Preprocess: {pre_time:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(draw, f"Inference: {inf_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(draw, f"Postprocess: {post_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 保存预测结果
    output_path = args.output_path if args.output_path else f"output_{Path(img_path).name}"
    cv2.imwrite(output_path, draw)
    print(f"Prediction result saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlc-model', type=str, required=True, help='DLC model file')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-path', type=str, default=None, help='Path to save output image (optional)')
    parser.add_argument('--runtime', type=str, choices=['cpu', 'gpu', 'dsp'], default='cpu', help='Runtime backend (cpu, gpu, or dsp)')
    parser.add_argument('--H', type=int, default=480, help='推理尺寸H')
    parser.add_argument('--W', type=int, default=640, help='推理尺寸W')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_image(args)

# python3 test_dlc.py --dlc-model yolov8_int8_480_640_Q.dlc --image-path data/bus.jpg --output-path outputs/bus_dlc_int8_480_640_Q.jpg --H 480 --W 640
# python3 test_dlc.py --dlc-model yolov8_int8_320_320_Q.dlc --image-path data/bus.jpg --output-path outputs/bus_dlc_int8_320_320_Q.jpg --H 320 --W 320


