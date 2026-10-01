from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_RAW = ROOT / "glukometr"
IMAGES_RAW = DATASET_RAW / "images"         
ANNOTATIONS_RAW = DATASET_RAW / "annotations" 

# --- датасет (YOLO-ready) ---
DATASET_YOLO = ROOT / "dataset"
IMAGES_TRAIN = DATASET_YOLO / "images" / "train"
IMAGES_VAL = DATASET_YOLO / "images" / "val"
LABELS_TRAIN = DATASET_YOLO / "labels" / "train"
LABELS_VAL = DATASET_YOLO / "labels" / "val"
DATA_YAML = DATASET_YOLO / "data.yaml"

# --- классы ---
CLASSES = ["display"]  
CLASSES = ["glucometer"]
VAL_SPLIT = 0.15       
SEED = 42

# --- обучение ---
YOLO_BASE = "yolo11n.pt" 
EPOCHS = 100
IMGSZ = 640
BATCH = 16
PROJECT_DIR = ROOT / "runs"
RUN_NAME = "glucometer_display"

BEST_WEIGHTS = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
DET_CONF = 0.4