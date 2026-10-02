"""
Тестирование обученной YOLO-модели без OCR.

Использование:
    # одно изображение (печатает боксы, сохраняет визуализацию)
    python test_yolo.py path/to/image.jpg

    # одно изображение с сохранением картинки
    python test_yolo.py path/to/image.jpg --save-vis result.jpg

    # папка с изображениями + CSV
    python test_yolo.py path/to/folder --save-csv detections.csv

    # с другим порогом уверенности
    python test_yolo.py image.jpg --conf 0.6

    # конкретные веса
    python test_yolo.py image.jpg --weights runs/detect/train2/weights/best.pt

    # без GPU
    python test_yolo.py image.jpg --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------- конфигурация по умолчанию ----------
DEFAULT_WEIGHTS = Path("runs/glucometer_display/weights/best.pt")
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.45
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


# ---------- загрузка модели ----------
_model_cache = {}


def get_model(weights: Path, device: str | None = None):
    key = (str(weights), device)
    if key not in _model_cache:
        if not weights.exists():
            raise SystemExit(f"Не найдены веса: {weights}")
        print(f"[i] Загрузка модели: {weights}")
        model = YOLO(str(weights))
        if device:
            model.to(device)
        _model_cache[key] = model
    return _model_cache[key]


# ---------- утилиты ----------
def fmt_conf(c: float) -> str:
    return f"{c:.3f}"


def detect_image(model: YOLO, img_path: Path, conf: float, iou: float):
    """
    Возвращает список объектов: [{cls, cls_id, conf, bbox=(x1,y1,x2,y2)}, ...]
    """
    results = model(str(img_path), conf=conf, iou=iou, verbose=False)
    r = results[0]

    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets, r

    names = r.names  # {id: name}
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    for box, cid, c in zip(xyxy, cls_ids, confs):
        x1, y1, x2, y2 = map(int, box.tolist())
        dets.append({
            "cls_id": int(cid),
            "cls": names.get(int(cid), str(cid)),
            "conf": float(c),
            "bbox": (x1, y1, x2, y2),
        })

    # сортировка по уверенности (сначала самые уверенные)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets, r


def save_visualization(img_path: Path, dets, out_path: Path):
    """Рисует bbox и подписи поверх копии изображения."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Не удалось прочитать {img_path}")
        return

    h, w = img.shape[:2]
    thickness = max(2, int(min(h, w) / 400))
    font_scale = max(0.6, min(h, w) / 1200)

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"{d['cls']} {fmt_conf(d['conf'])}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # фон под текст
        cv2.rectangle(
            img,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw + 6, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[i] Визуализация сохранена: {out_path}")


# ---------- CLI ----------
def process_single(model, img_path: Path, conf, iou, save_vis: Path | None):
    print(f"\n=== {img_path} ===")
    dets, _ = detect_image(model, img_path, conf, iou)

    if not dets:
        print("  Ничего не найдено (попробуйте --conf ниже)")
        return dets

    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = d["bbox"]
        print(
            f"  [{i}] cls={d['cls']:<15} conf={fmt_conf(d['conf'])} "
            f"bbox=({x1},{y1},{x2},{y2}) w={x2-x1} h={y2-y1}"
        )

    if save_vis:
        save_visualization(img_path, dets, save_vis)

    return dets


def process_folder(model, folder: Path, conf, iou, save_csv: Path | None,
                   save_vis_dir: Path | None):
    images = sorted(p for p in folder.iterdir() if p.suffix in IMG_EXTS)
    if not images:
        raise SystemExit(f"В папке нет изображений: {folder}")

    rows = []
    total = 0
    for img_path in images:
        dets, _ = detect_image(model, img_path, conf, iou)
        n = len(dets)
        total += n
        if n == 0:
            print(f"{img_path.name}: ничего не найдено")
        else:
            top = dets[0]
            print(
                f"{img_path.name}: найдено {n}, "
                f"top={top['cls']} conf={fmt_conf(top['conf'])}"
            )
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            rows.append({
                "file": img_path.name,
                "cls": d["cls"],
                "conf": f"{d['conf']:.4f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        if save_vis_dir:
            save_visualization(img_path, dets, save_vis_dir / img_path.name)

    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with save_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["file", "cls", "conf", "x1", "y1", "x2", "y2"]
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\n[i] CSV сохранён: {save_csv}")

    print(f"\n[i] Обработано изображений: {len(images)}, "
          f"всего детекций: {total}")


def main():
    ap = argparse.ArgumentParser(
        description="Тест обученной YOLO-модели (без OCR)."
    )
    ap.add_argument("input", help="Путь к изображению или папке")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help=f"Путь к .pt весам (по умолчанию {DEFAULT_WEIGHTS})")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF,
                    help=f"Порог уверенности (по умолчанию {DEFAULT_CONF})")
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU,
                    help=f"NMS IoU (по умолчанию {DEFAULT_IOU})")
    ap.add_argument("--device", default=None,
                    help='Устройство: "0" (GPU), "cpu" (по умолчанию — авто)')
    ap.add_argument("--save-vis", type=Path, default=None,
                    help="Сохранить визуализацию (для одного изображения)")
    ap.add_argument("--save-vis-dir", type=Path, default=None,
                    help="Сохранить визуализации для папки")
    ap.add_argument("--save-csv", type=Path, default=None,
                    help="CSV с детекциями (для папки)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Не найдено: {src}")

    model = get_model(args.weights, args.device)

    if src.is_file():
        process_single(model, src, args.conf, args.iou, args.save_vis)
    elif src.is_dir():
        process_folder(
            model, src, args.conf, args.iou,
            args.save_csv, args.save_vis_dir,
        )
    else:
        raise SystemExit(f"Не изображение и не папка: {src}")


if __name__ == "__main__":
    sys.exit(main())