# ==========================================================
#  config.py - Cau hinh huan luyen LoRA cho MD_OCR
#  Chinh tat ca tham so tai day, khong can sua file train.py
# ==========================================================

# --- Mo hinh goc ---
MODEL_ID = "xiaomi-research/Xiaomi-OCR-0"

# --- Thu muc du lieu (dat bien moi truong MD_OCR_BASE hoac chinh truc tiep) ---
import os
from pathlib import Path

BASE_DIR    = Path(os.environ.get("MD_OCR_BASE", "."))
DATASET_DIR = BASE_DIR / "dataset_date_md"       # thu muc chua train.jsonl / val.jsonl / train/
OUTPUT_DIR  = BASE_DIR / "output_lora"            # noi luu adapter va checkpoint
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"         # thu muc adapter cuoi cung

# --- Warm-start ---
# Dat WARM_START_ADAPTER = None de train tu dau (fresh start).
# Dat duong dan toi thu muc adapter de khoi dong tu checkpoint co san.
WARM_START_ADAPTER = None
# Vi du warm-start:
# WARM_START_ADAPTER = BASE_DIR / "lora_adapter_v9"

# --- Sieu tham so LoRA ---
LORA_R         = 32
LORA_ALPHA     = 32      # ty le alpha/r = 1.0 (on dinh)
LORA_DROPOUT   = 0.05
INCLUDE_VISION = True    # True: ap dung LoRA ca visual encoder

# --- Huan luyen ---
NUM_EPOCHS     = 5
LR             = 3e-5
LR_SCHEDULER   = "cosine"
WARMUP_STEPS   = 30
WEIGHT_DECAY   = 0.01
TARGET_EFF_BS  = 64      # effective batch size muc tieu
PER_DEVICE_BS  = 4       # batch size thuc te moi GPU/CPU (giam neu het RAM)
GRAD_ACCUM     = max(1, TARGET_EFF_BS // PER_DEVICE_BS)  # tu tinh

# --- Prompt ---
SHORT_PROMPT = "The text in this image is a handwritten date in month/day format. Transcribe it. Output only the text."

# --- Gioi han thoi gian (gio) - dat None de khong gioi han ---
TIME_LIMIT_H = None

# --- Khac ---
SEED          = 20261116
LOG_STEPS     = 10       # in log sau moi bao nhieu step
EVAL_STEPS    = 100      # danh gia sau moi bao nhieu step (None = cuoi moi epoch)
SAVE_STEPS    = 100      # luu checkpoint sau moi bao nhieu step
PIN_TRANSFORMERS = "5.17.0"  # None de dung phien ban moi nhat

