from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from utils.db_utils import SegDetectorRepresenter
from utils.imgproc_utils import letterbox
from utils.log import LOG
from utils.textblock import TextBlock, group_output, visualize_textblocks
from utils.yolo_nms import postprocess_yolo


@dataclass
class DetectionResult:
    mask: np.ndarray
    text_blocks: list[TextBlock] = field(default_factory=list)
    original_shape: tuple[int, int] = (0, 0)


class ComicTextDetector:
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "comic-text-detector.onnx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        input_size: int = 1024,
        conf_thresh: float = 0.4,
        nms_thresh: float = 0.35,
    ) -> None:
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.providers = providers or ["CPUExecutionProvider"]
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )
        LOG.info("[detector] Model loaded (%s)", self.session.get_providers()[0])

        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_shape = (self.input_size, self.input_size)
        img_in, _, (dw, dh) = letterbox(rgb, new_shape=new_shape, auto=False, stride=64)
        blob = img_in.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        return np.expand_dims(blob, axis=0), int(dw), int(dh)

    def _postprocess_mask(self, seg: np.ndarray, dw: int, dh: int, im_w: int, im_h: int) -> np.ndarray:
        mask = np.squeeze(seg)
        mask = (mask * 255).astype(np.uint8)
        mask = mask[: mask.shape[0] - dh, : mask.shape[1] - dw]
        return cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

    def detect(self, image: np.ndarray) -> DetectionResult:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        im_h, im_w = image.shape[:2]
        blob, dw, dh = self._preprocess(image)

        blk_raw, seg_raw, det_raw = self.session.run(None, {self.input_name: blob})

        resize_ratio = (
            im_w / (self.input_size - dw),
            im_h / (self.input_size - dh),
        )
        blines, cls, confs = postprocess_yolo(
            blk_raw, self.conf_thresh, self.nms_thresh, resize_ratio
        )

        lines, scores = self.seg_rep(det_raw)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)[0]
        line_boxes = lines[0][idx]
        line_scores = scores[0][idx]

        mask = self._postprocess_mask(seg_raw, dw, dh, im_w, im_h)

        if line_boxes.size == 0:
            decoded_lines: list = []
        else:
            line_boxes = line_boxes.astype(np.float64)
            line_boxes[..., 0] *= resize_ratio[0]
            line_boxes[..., 1] *= resize_ratio[1]
            decoded_lines = line_boxes.astype(np.int32)

        blk_list = group_output((blines, cls, confs), decoded_lines, im_w, im_h, mask)

        return DetectionResult(
            mask=mask,
            text_blocks=blk_list,
            original_shape=(im_h, im_w),
        )

    def visualize(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        vis = image.copy()
        return visualize_textblocks(vis, result.text_blocks)
