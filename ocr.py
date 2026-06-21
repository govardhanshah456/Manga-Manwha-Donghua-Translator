from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from PIL import Image

from utils.textblock import TextBlock


class MediaType(str, Enum):
    MANGA = "manga"
    MANHWA = "manhwa"
    DONGHUA = "donghua"
    MANHUA = "manhua"


def extract_block_image(
    image: np.ndarray,
    block: TextBlock,
    padding_ratio: float = 0.12,
    min_size: int = 32,
) -> np.ndarray | None:
    x1, y1, x2, y2 = block.xyxy
    h, w = image.shape[:2]
    pad_x = max(int((x2 - x1) * padding_ratio), 2)
    pad_y = max(int((y2 - y1) * padding_ratio), 2)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]
    if crop_h < min_size or crop_w < min_size:
        scale = max(min_size / crop_h, min_size / crop_w)
        crop = cv2.resize(
            crop,
            (int(round(crop_w * scale)), int(round(crop_h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def _split_recognized_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    return [line for line in lines if line]


class ComicTextRecognizer:
    """Recognize text in detected comic text blocks."""

    def __init__(
        self,
        media_type: MediaType = MediaType.MANGA,
        verbose: bool = True,
    ) -> None:
        self.media_type = media_type
        self.verbose = verbose
        self._manga_ocr = None
        self._rapid_engines: dict[str, object] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _get_manga_ocr(self):
        if self._manga_ocr is None:
            self._log("[ocr] Loading manga-ocr (ONNX)...")
            from manga_ocr import MangaOcr

            self._manga_ocr = MangaOcr()
        return self._manga_ocr

    def _get_rapid_ocr(self, lang_key: str):
        if lang_key not in self._rapid_engines:
            from rapidocr import EngineType, LangRec, OCRVersion, RapidOCR

            lang_map = {
                "en": LangRec.EN,
                "ch": LangRec.CH,
                "ko": LangRec.KOREAN,
                "ja": LangRec.JAPAN,
            }
            lang_type = lang_map.get(lang_key, LangRec.CH)
            self._log(f"[ocr] Loading RapidOCR ({lang_key})...")
            self._rapid_engines[lang_key] = RapidOCR(
                params={
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Cls.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.lang_type": lang_type,
                    "Rec.ocr_version": OCRVersion.PPOCRV4,
                }
            )
        return self._rapid_engines[lang_key]

    def _backend_for_block(self, block: TextBlock) -> str:
        if block.language == "ja":
            return "manga"
        if block.language == "eng":
            return "rapid_en"

        if self.media_type == MediaType.MANHWA:
            return "rapid_ko"
        if self.media_type in (MediaType.DONGHUA, MediaType.MANHUA):
            return "rapid_ch"
        return "manga"

    def _recognize_manga(self, crop: np.ndarray) -> list[str]:
        mocr = self._get_manga_ocr()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        text = mocr(Image.fromarray(rgb))
        return _split_recognized_text(text)

    def _recognize_rapid(self, crop: np.ndarray, lang_key: str) -> list[str]:
        engine = self._get_rapid_ocr(lang_key)
        output = engine(crop)
        if output is None or output.txts is None:
            return []
        return [text.strip() for text in output.txts if text and text.strip()]

    def recognize_block(self, image: np.ndarray, block: TextBlock) -> list[str]:
        crop = extract_block_image(image, block)
        if crop is None:
            return []

        backend = self._backend_for_block(block)
        if backend == "manga":
            return self._recognize_manga(crop)
        if backend == "rapid_en":
            return self._recognize_rapid(crop, "en")
        if backend == "rapid_ko":
            return self._recognize_rapid(crop, "ko")
        if backend == "rapid_ch":
            return self._recognize_rapid(crop, "ch")
        return self._recognize_manga(crop)

    def recognize(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[TextBlock]:
        self._log(f"[ocr] Recognizing {len(text_blocks)} text blocks...")
        for block in text_blocks:
            block.text = self.recognize_block(image, block)
        return text_blocks
