from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray

# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')

# angle scale
ANGLE_SCALE = 1 / np.pi * 180.0


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images


def crop_mask(masks: ndarray, bboxes: ndarray) -> ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3],
                              1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def box_iou(box1: ndarray, box2: ndarray) -> float:
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2
    x1 = max(x11, x12)
    y1 = max(y11, y12)
    x2 = min(x21, x22)
    y2 = min(y21, y22)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x21 - x11) * (y21 - y11) + (x22 - x12) * (y22 -
                                                            y12) - inter_area
    return max(0, inter_area / union_area)


def NMSBoxes(boxes: ndarray,
             scores: ndarray,
             labels: ndarray,
             iou_thres: float,
             agnostic: bool = False):
    num_boxes = boxes.shape[0]
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    labels = labels[order]

    indices = []

    for i in range(num_boxes):
        box_a = boxes[i]
        label_a = labels[i]
        keep = True
        for j in indices:
            box_b = boxes[j]
            label_b = labels[j]
            if not agnostic and label_a != label_b:
                continue
            if box_iou(box_a, box_b) > iou_thres:
                keep = False
        if keep:
            indices.append(i)

    indices = np.array(indices, dtype=np.int32)
    return order[indices]


def det_postprocess(data: Tuple[ndarray, ndarray, ndarray, ndarray]):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    # print(nums)
    if nums == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(
            (0, ), dtype=np.float32), np.empty((0, ), dtype=np.int32)
    # check score negative
    scores[scores < 0] = 1 + scores[scores < 0]
    # print(bboxes.shape)
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels


def det_postprocess_nms(pred: np.ndarray, conf_thres: float = 0.5, iou_thres: float = 0.5, agnostic: bool = False):
    """
    YOLOv8 后处理：从 pred 张量中提取 boxes、conf、class_ids，应用置信度过滤和 NMS。
    参数:
        pred: np.ndarray, 形状 [1, N, 84]，包含 [x1, y1, x2, y2, score1, ..., score80]
        conf_thres: float，置信度阈值
        iou_thres: float，NMS 的 IoU 阈值
        agnostic: bool，是否执行类无关 NMS
    返回:
        bboxes: np.ndarray, 形状 [N', 4]，类型 np.float32，边界框 [x1, y1, x2, y2]
        scores: np.ndarray, 形状 [N']，类型 np.float32，置信度
        labels: np.ndarray, 形状 [N']，类型 np.int32，类别索引
        如果没有检测框，返回空数组
    """
    batch_size = pred.shape[0]
    assert batch_size == 1, "Only single batch (B=1) is supported to match det_postprocess"

    # 提取 boxes 和 scores
    boxes = pred[0, :, :4]  # [N, 4]，[x1, y1, x2, y2]
    scores = pred[0, :, 4:] # [N, 80]，类别分数

    # 计算置信度和类别索引
    conf = np.max(scores, axis=1)           # [N]，最大置信度
    class_ids = np.argmax(scores, axis=1)   # [N]，类别索引

    # 置信度过滤
    mask = conf > conf_thres
    if not np.any(mask):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )

    boxes = boxes[mask]         # [N', 4]
    conf = conf[mask]           # [N']
    class_ids = class_ids[mask] # [N']

    # 应用 NMS
    keep_indices = NMSBoxes(boxes, conf, class_ids, iou_thres, agnostic)

    # 如果没有保留框，返回空数组
    if keep_indices.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )

    # 提取过滤后的结果
    bboxes = boxes[keep_indices]         # [N'', 4]
    scores = conf[keep_indices]          # [N'']
    labels = class_ids[keep_indices]     # [N'']
    return bboxes, scores, labels


def det_postprocess_nms_cv2(pred: np.ndarray, conf_thres: float = 0.5, iou_thres: float = 0.5, agnostic: bool = False):
    """
    YOLOv8 后处理：从 pred 张量中提取 boxes、conf、class_ids，应用置信度过滤和 NMS。
    参数:
        pred: np.ndarray, 形状 [1, N, 84]，包含 [x1, y1, x2, y2, score1, ..., score80]
        conf_thres: float，置信度阈值
        iou_thres: float，NMS 的 IoU 阈值
        agnostic: bool，是否执行类无关 NMS
    返回:
        bboxes: np.ndarray, 形状 [N', 4]，类型 np.float32，边界框 [x1, y1, x2, y2]
        scores: np.ndarray, 形状 [N']，类型 np.float32，置信度
        labels: np.ndarray, 形状 [N']，类型 np.int32，类别索引
        如果没有检测框，返回空数组
    """
    batch_size = pred.shape[0]
    assert batch_size == 1, "Only single batch (B=1) is supported to match det_postprocess"

    # 提取 boxes 和 scores
    boxes = pred[0, :, :4]  # [N, 4]，[x1, y1, x2, y2]
    scores = pred[0, :, 4:] # [N, 80]，类别分数

    # 计算置信度和类别索引
    conf = np.max(scores, axis=1)           # [N]，最大置信度
    class_ids = np.argmax(scores, axis=1)   # [N]，类别索引

    # 置信度过滤
    mask = conf > conf_thres
    if not np.any(mask):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )

    boxes = boxes[mask]         # [N', 4]
    conf = conf[mask]           # [N']
    class_ids = class_ids[mask] # [N']

    # 转换为 cv2.dnn.NMSBoxes 所需格式
    # boxes 需要转换为 [x, y, w, h] 格式
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1

    # 准备输入数据
    indices = []
    if agnostic:
        # 类无关 NMS：所有框视为同一类
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), conf.tolist(), conf_thres, iou_thres
        )
    else:
        # 类相关 NMS：按类别分组
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes_xywh[cls_mask]
            cls_scores = conf[cls_mask]
            if len(cls_boxes) > 0:
                cls_indices = cv2.dnn.NMSBoxes(
                    cls_boxes.tolist(), cls_scores.tolist(), conf_thres, iou_thres
                )
                # 调整索引以匹配原始数组
                cls_indices = np.where(cls_mask)[0][cls_indices]
                indices.extend(cls_indices.tolist())
        indices = np.array(indices)

    # 如果没有保留框，返回空数组
    if indices.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )

    # 提取过滤后的结果
    bboxes = boxes[indices]         # [N'', 4]
    scores = conf[indices]          # [N'']
    labels = class_ids[indices]     # [N'']
    return bboxes, scores, labels
