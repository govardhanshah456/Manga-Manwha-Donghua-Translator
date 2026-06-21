import json
import sys
from pathlib import Path

import cv2

from detector import ComicTextDetector
from ocr import ComicTextRecognizer, MediaType


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path("input/manga_input_1.jpg")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    print("=== Comic Text Detector + OCR ===\n")

    detector = ComicTextDetector()
    result = detector.detect(image)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Detected text blocks: {len(result.text_blocks)}\n")

    recognizer = ComicTextRecognizer(media_type=MediaType.MANGA)
    recognizer.recognize(image, result.text_blocks)

    ocr_entries = []
    for i, blk in enumerate(result.text_blocks):
        text = blk.get_text()
        vertical = "vertical" if blk.vertical else "horizontal"
        print(f"  [{i}] {blk.language} ({vertical}): {text or '(empty)'}")
        ocr_entries.append(
            {
                "index": i,
                "language": blk.language,
                "vertical": bool(blk.vertical),
                "bbox": [int(x) for x in blk.xyxy],
                "lines": len(blk.lines),
                "text": blk.text,
                "text_joined": text,
            }
        )

    vis = detector.visualize(image, result)
    mask_path = output_dir / "mask.png"
    vis_path = output_dir / "detected.jpg"
    ocr_path = output_dir / "ocr.json"

    cv2.imwrite(str(mask_path), result.mask)
    cv2.imwrite(str(vis_path), vis)
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_entries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved mask to {mask_path}")
    print(f"Saved visualization to {vis_path}")
    print(f"Saved OCR results to {ocr_path}")


if __name__ == "__main__":
    main()
