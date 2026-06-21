import numpy as np


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = np.maximum(box1[:, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, 3], box2[:, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    return inter / (area1 + area2 - inter + 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou = _box_iou(boxes[i : i + 1], boxes[order[1:]])
        order = order[1:][iou[0] <= iou_thres]
    return np.array(keep, dtype=np.int64)


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[np.ndarray]:
    if prediction.ndim == 2:
        prediction = prediction[np.newaxis, ...]

    output = []
    max_wh = 4096

    for pred in prediction:
        xc = pred[..., 4] > conf_thres
        x = pred[xc]
        if x.shape[0] == 0:
            output.append(np.zeros((0, 6)))
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf = x[:, 5:].max(axis=1)
        cls = x[:, 5:].argmax(axis=1)
        x = np.column_stack([box, conf, cls])
        x = x[conf > conf_thres]

        if x.shape[0] == 0:
            output.append(np.zeros((0, 6)))
            continue

        if x.shape[0] > 30000:
            order = x[:, 4].argsort()[::-1][:30000]
            x = x[order]

        c = x[:, 5:6] * max_wh
        boxes = x[:, :4] + c
        scores = x[:, 4]
        keep = _nms(boxes, scores, iou_thres)
        if len(keep) > max_det:
            keep = keep[:max_det]
        output.append(x[keep])

    return output


def postprocess_yolo(
    det: np.ndarray,
    conf_thresh: float,
    nms_thresh: float,
    resize_ratio: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    if det.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs
