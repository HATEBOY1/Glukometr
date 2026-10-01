from ultralytics import YOLO

import config as cfg


def main():
    if not cfg.DATA_YAML.exists():
        raise SystemExit(f"Нет {cfg.DATA_YAML}. Сначала запустите voc2yolo.py")

    model = YOLO(cfg.YOLO_BASE)

    model.train(
        data=str(cfg.DATA_YAML),
        epochs=cfg.EPOCHS,
        imgsz=cfg.IMGSZ,
        batch=cfg.BATCH,
        patience=20,
        # аугментации — числа НЕ отражаем
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        # прочее
        project=str(cfg.PROJECT_DIR),
        name=cfg.RUN_NAME,
        exist_ok=True,
        seed=cfg.SEED,
        verbose=True,
        plots=True,
    )

    print(f"\n[DONE] Веса: {cfg.BEST_WEIGHTS}")


if __name__ == "__main__":
    main()