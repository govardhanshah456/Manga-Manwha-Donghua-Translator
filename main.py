import json
import logging
import os
import warnings
from pathlib import Path

import cv2

from utils.log import LOG, setup_logging, verbose_blocks

warnings.filterwarnings("ignore", module="huggingface_hub")

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)

from detector import ComicTextDetector
from ocr import ComicTextRecognizer, MediaType
from translator import GeminiTranslator


def main() -> None:
    setup_logging()

    input_path = Path("input/manga_input_1.jpg")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    target_language = "English"

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    LOG.info("=== Comic Translator ===")
    LOG.info("Detecting text...")

    detector = ComicTextDetector()
    result = detector.detect(image)
    LOG.info("Found %d text blocks (%dx%d)", len(result.text_blocks), image.shape[1], image.shape[0])

    LOG.info("Running OCR...")
    recognizer = ComicTextRecognizer(media_type=MediaType.MANGA)
    recognizer.recognize(image, result.text_blocks)
    LOG.info("OCR done.")

    LOG.info("Translating to %s...", target_language)
    translator = GeminiTranslator(target_language=target_language)
    translator.translate_blocks(result.text_blocks)
    LOG.info("Translation done.")

    entries = []
    for i, blk in enumerate(result.text_blocks):
        source = blk.get_text()
        translation = blk.translation or ""
        if verbose_blocks():
            LOG.info("  [%d] %s: %s → %s", i, blk.language, source or "(empty)", translation or "(empty)")

        entries.append(
            {
                "index": i,
                "language": blk.language,
                "vertical": bool(blk.vertical),
                "bbox": [int(x) for x in blk.xyxy],
                "lines": len(blk.lines),
                "text": blk.text,
                "text_joined": source,
                "translation": translation,
                "target_language": target_language,
            }
        )

    vis = detector.visualize(image, result)
    mask_path = output_dir / "mask.png"
    vis_path = output_dir / "detected.jpg"
    results_path = output_dir / "translations.json"

    cv2.imwrite(str(mask_path), result.mask)
    cv2.imwrite(str(vis_path), vis)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    LOG.info("Saved %s", mask_path)
    LOG.info("Saved %s", vis_path)
    LOG.info("Saved %s", results_path)


if __name__ == "__main__":
    main()
