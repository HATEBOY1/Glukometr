# debug_voc.py — положите в backend/
import xml.etree.ElementTree as ET
from pathlib import Path
import config as cfg

xmls = list(cfg.ANNOTATIONS_RAW.glob("*.xml"))
print(f"Найдено XML: {len(xmls)}")
print(f"CLASSES в config.py: {cfg.CLASSES}")
print(f"ANNOTATIONS_RAW: {cfg.ANNOTATIONS_RAW}")
print(f"IMAGES_RAW:      {cfg.IMAGES_RAW}")
print("-" * 60)

# берём первые 3 файла
for x in xmls[:3]:
    print(f"\n>>> {x.name}")
    root = ET.parse(x).getroot()

    size = root.find("size")
    if size is None:
        print("  !! нет <size>")
        continue
    w = size.find("width").text
    h = size.find("height").text
    print(f"  size: {w} x {h}")

    objs = root.findall("object")
    print(f"  objects: {len(objs)}")

    for o in objs:
        name = o.find("name").text.strip()
        b = o.find("bndbox")
        xmin = b.find("xmin").text
        ymin = b.find("ymin").text
        xmax = b.find("xmax").text
        ymax = b.find("ymax").text
        in_classes = name in cfg.CLASSES
        print(f"    name={name!r}  bbox=({xmin},{ymin},{xmax},{ymax})  in CLASSES? {in_classes}")

# сколько изображений нашлось под каждый XML
print("\n" + "=" * 60)
missing = 0
for x in xmls:
    stem = x.stem
    found = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = cfg.IMAGES_RAW / f"{stem}{ext}"
        if p.exists():
            found = p
            break
    if found is None:
        missing += 1
        if missing <= 5:
            print(f"  НЕТ изображения для {stem}")
print(f"Всего XML: {len(xmls)}, без картинки: {missing}")