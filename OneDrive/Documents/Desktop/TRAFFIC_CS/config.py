from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
VIDEO_DIR = DATA_DIR / "video_files"
HISTORY_FILE = DATA_DIR / "history"

OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_NAME = ROOT_DIR / "models" / "yolov8n.pt"

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

SIGNAL_RULES = [
    (0, 5, "LOW", 15),
    (6, 15, "MEDIUM", 30),
    (16, 30, "HIGH", 45),
]

for folder in [
    IMAGE_DIR,
    VIDEO_DIR,
    OUTPUT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)