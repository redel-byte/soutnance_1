import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("STUDENT_DATA_PATH", ROOT / "data/raw/student-mental.csv"))
OUTPUT_DIR = Path(os.environ.get("STUDENT_OUTPUT_DIR", ROOT / "artifacts"))

TARGET = "Mental_Health_Risk"
RANDOM_STATE = 22
Z_SCORE_THRESHOLD = 3

