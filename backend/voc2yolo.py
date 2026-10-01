"""
Конвертация Pascal VOC XML → YOLO txt + сборка датасета (train/val).
Запуск:  python voc2yolo.py
"""
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

import config as cfg


def parse_voc(xml_path: Path):
    """Возвращает (width, height, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Нет <size> в {xml_path}")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # clip
        xmin = max(0.0, min(xmin, w))
        ymin = max(0.0, min(ymin, h))
        xmax = max(0.0, min(xmax, w))
        ymax = max(0.0, min(ymax, h))

        # игнорируем слишком маленькие / битые
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue

        objs.append((name, xmin, ymin, xmax, ymax))
    return w, h, objs


def to_yolo_line(w, h, class_name, xmin, ymin, xmax, ymax):
    cls_id = cfg.CLASSES.index(class_name)
    xc = ((xmin + xmax) / 2) / w
    yc = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def find_image(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = cfg.IMAGES_RAW / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def ensure_dirs():
    for d in (cfg.IMAGES_TRAIN, cfg.IMAGES_VAL, cfg.LABELS_TRAIN, cfg.LABELS_VAL):
        d.mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    cfg.DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"path: {cfg.DATASET_YOLO.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(cfg.CLASSES)}\n"
        f"names: {cfg.CLASSES}\n"
    )
    cfg.DATA_YAML.write_text(text, encoding="utf-8")
    print(f"[OK] Записан {cfg.DATA_YAML}")


def main():
    if not cfg.ANNOTATIONS_RAW.exists():
        raise SystemExit(f"Нет папки с XML: {cfg.ANNOTATIONS_RAW}")

    xml_files = sorted(cfg.ANNOTATIONS_RAW.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"XML не найдены в {cfg.ANNOTATIONS_RAW}")

    ensure_dirs()

    # стабильный split
    random.seed(cfg.SEED)
    random.shuffle(xml_files)
    n_val = max(1, int(len(xml_files) * cfg.VAL_SPLIT))
    val_set = set(xml_files[:n_val])

    skipped = 0
    for xml_path in tqdm(xml_files, desc="Конвертация"):
        stem = xml_path.stem
        img_path = find_image(stem)
        if img_path is None:
            skipped += 1
            continue

        try:
            w, h, objs = parse_voc(xml_path)
        except Exception as e:
            print(f"[skip] {xml_path.name}: {e}")
            skipped += 1
            continue

        lines = []
        for name, xmin, ymin, xmax, ymax in objs:
            if name not in cfg.CLASSES:
                # неизвестный класс — пропускаем (или расширьте CLASSES)
                continue
            lines.append(to_yolo_line(w, h, name, xmin, ymin, xmax, ymax))

        is_val = xml_path in val_set
        img_dst_dir = cfg.IMAGES_VAL if is_val else cfg.IMAGES_TRAIN
        lbl_dst_dir = cfg.LABELS_VAL if is_val else cfg.LABELS_TRAIN

        shutil.copy2(img_path, img_dst_dir / img_path.name)
        (lbl_dst_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    write_data_yaml()
    print(f"[DONE] Изображений обработано: {len(xml_files) - skipped}, пропущено: {skipped}")


if __name__ == "__main__":
    main()