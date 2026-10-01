import argparse
import csv
from pathlib import Path

import pandas as pd

import config as cfg
from predict import read_value


def load_gt(csv_path: Path) -> dict:
    gt = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["file"]] = float(row["value"])
    return gt


def evaluate(images_dir: Path, gt: dict, conf: float = cfg.DET_CONF):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    files = [p for p in sorted(images_dir.iterdir()) if p.suffix in exts]

    rows = []
    det_fail = 0
    ocr_fail = 0

    for p in files:
        if p.name not in gt:
            continue
        r = read_value(str(p), conf=conf)
        pred = r["value"]
        true = gt[p.name]

        exact = (pred is not None) and (abs(pred - true) < 1e-6)
        abs_err = abs(pred - true) if pred is not None else None

        if r["bbox"] is None:
            det_fail += 1
        if pred is None and r["bbox"] is not None:
            ocr_fail += 1

        rows.append({
            "file": p.name,
            "true": true,
            "pred": pred,
            "raw_ocr": r["raw_ocr"],
            "abs_err": abs_err,
            "exact": exact,
            "det_conf": r["det_conf"],
        })

    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        print("Нет данных для оценки.")
        return df

    exact_match = df["exact"].mean()
    mae = df["abs_err"].dropna().mean() if df["abs_err"].notna().any() else float("nan")
    det_fail_rate = det_fail / n
    ocr_fail_rate = ocr_fail / n

    print("=" * 60)
    print(f"Всего изображений:      {n}")
    print(f"Exact match:            {exact_match:.2%}")
    print(f"MAE (по распознанным):  {mae:.4f}")
    print(f"Провал детекции:        {det_fail_rate:.2%}")
    print(f"Провал OCR (число):     {ocr_fail_rate:.2%}")
    print("=" * 60)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=cfg.DET_CONF)
    ap.add_argument("--out", type=Path, default=None,
                    help="Куда сохранить отчёт CSV")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    df = evaluate(args.images, gt, conf=args.conf)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"[OK] Отчёт: {args.out}")


if __name__ == "__main__":
    main()