import os
os.environ["TORCH_DISABLE_SHM"] = "1"
import argparse
import csv
import re
from pathlib import Path
import sys
from pathlib import Path
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
from pathlib import Path as _P

import config as cfg
from preprocess import crop_with_padding, preprocess_for_ocr


_detector = None
_ocr = None



if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
    # то же для torchvision, если есть
    tv_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torchvision"
    if tv_lib.exists():
        try:
            os.add_dll_directory(str(tv_lib))
        except Exception:
            pass




if sys.platform == "win32":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for lib_dir in (site_packages / "torch" / "lib", site_packages / "torchvision"):
        if lib_dir.exists():
            try:
                os.add_dll_directory(str(lib_dir))
            except Exception:
                pass


def get_detector():
    global _detector
    if _detector is None:
        if not cfg.BEST_WEIGHTS.exists():
            raise SystemExit(
                f"Нет весов {cfg.BEST_WEIGHTS}. Сначала запустите train.py"
            )
        _detector = YOLO(str(cfg.BEST_WEIGHTS))
    return _detector


def get_ocr():
    global _ocr
    if _ocr is None:
        # use_angle_cls=True нужен, если изображения могут быть повёрнуты
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr


def clean_ocr_text(text: str) -> str:
    """Исправляем типичные ошибки OCR на цифровых дисплеях."""
    repl = {
        ",": ".",
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "l": "1", "I": "1", "|": "1", "!": "1",
        "S": "5", "s": "5",
        "B": "8",
        "g": "9", "q": "9",
        "Z": "2", "z": "2",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def extract_number(text: str):
    text = clean_ocr_text(text)
    # ищем первое число (целое или с точкой)
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def run_ocr(crop_bgr):
    proc = preprocess_for_ocr(crop_bgr)
    result = get_ocr().ocr(proc, cls=True)
    texts = []
    if result and result[0]:
        for line in result[0]:
            t, score = line[1][0], float(line[1][1])
            texts.append((t, score))
    raw = " ".join(t for t, _ in texts)
    return raw, texts


def read_value(image_path: str, conf: float = cfg.DET_CONF):
    img = cv2.imread(str(image_path))
    if img is None:
        return {"value": None, "raw_ocr": "", "bbox": None,
                "det_conf": None, "ocr_items": []}

    results = get_detector()(str(image_path), conf=conf, verbose=False)[0]

    if len(results.boxes) == 0:
        return {"value": None, "raw_ocr": "", "bbox": None,
                "det_conf": None, "ocr_items": []}

    best = max(results.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    det_conf = float(best.conf)

    crop, box = crop_with_padding(img, (x1, y1, x2, y2), pad=5)
    if crop.size == 0:
        return {"value": None, "raw_ocr": "", "bbox": box,
                "det_conf": det_conf, "ocr_items": []}

    raw, items = run_ocr(crop)
    value = extract_number(raw)

    return {
        "value": value,
        "raw_ocr": raw,
        "bbox": box,
        "det_conf": det_conf,
        "ocr_items": items,
    }


def process_folder(folder: Path, save_csv: Path | None):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    files = [p for p in sorted(folder.iterdir()) if p.suffix in exts]
    rows = []
    for p in files:
        r = read_value(p)
        rows.append({
            "file": p.name,
            "value": r["value"],
            "raw_ocr": r["raw_ocr"],
            "det_conf": r["det_conf"],
        })
        print(f"{p.name}: value={r['value']} raw='{r['raw_ocr']}' det={r['det_conf']}")
    if save_csv:
        with save_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "value", "raw_ocr", "det_conf"])
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] CSV: {save_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Путь к изображению или папке")
    ap.add_argument("--conf", type=float, default=cfg.DET_CONF,
                    help="Порог уверенности детектора")
    ap.add_argument("--save-csv", type=Path, default=None,
                    help="Куда сохранить CSV (для папки)")
    args = ap.parse_args()

    p = Path(args.input)
    if p.is_dir():
        process_folder(p, args.save_csv)
    elif p.is_file():
        r = read_value(str(p), conf=args.conf)
        print(f"Файл: {p}")
        print(f"Значение: {r['value']}")
        print(f"Raw OCR:  '{r['raw_ocr']}'")
        print(f"BBox:     {r['bbox']}")
        print(f"Det conf: {r['det_conf']}")
        print(f"OCR items: {r['ocr_items']}")
    else:
        raise SystemExit(f"Не найдено: {p}")


if __name__ == "__main__":
    main()