#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_dataset.py
------------------------------------------------------------------------------------------
Sinh du lieu M/D (Month/Day) cho bai toan OCR nhan dang ngay thang tren Xiaomi-OCR-0.

Cach chay:
    export MD_OCR_BASE=/duong/dan/toi/thu_muc_assets
    python3 gen_dataset.py

Hoac truyen bien moi truong truc tiep:
    MD_OCR_BASE=/path/to/assets python3 gen_dataset.py

Thu muc assets phai chua:
    ETL_Project/output/characters/   (thu muc 0-9 chua anh PNG tu ETL-3)
    handwritten_gen_package/background_MD/        (anh nen 1.png - 6.png)
    handwritten_gen_package/handwritten_digits_2000/1/  (anh so 1 viet tay)
    user_drawn_samples/drawings.jsonl + images/   (anh nguoi dung ve)
    raw_handwritten_crops/crops_five_new/         (anh so 5 cat thuc te)

Tai assets tai: https://drive.google.com/file/d/1SdrTt4B7FZ9b6sc_ibY4fVh-d_vPcR5F/view?usp=share_link
------------------------------------------------------------------------------------------
"""

import os
import cv2
import json
import random
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------------
# CAU HINH DUONG DAN
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("MD_OCR_BASE", "."))

ETL_DIR     = BASE_DIR / "ETL_Project/output/characters"
D1_DIR      = BASE_DIR / "handwritten_gen_package/handwritten_digits_2000/1"
BG_DIR      = BASE_DIR / "handwritten_gen_package/background_MD"
USER_DIR    = BASE_DIR / "user_drawn_samples"
CROPS_5_DIR = BASE_DIR / "raw_handwritten_crops/crops_five_new"

OUT_DIR       = BASE_DIR / "dataset_date_md"
TRAIN_IMG_DIR = OUT_DIR / "train"
TRAIN_JSONL   = OUT_DIR / "train.jsonl"
VAL_JSONL     = OUT_DIR / "val.jsonl"
STATS_JSON    = OUT_DIR / "stats.json"

# ---------------------------------------------------------------------------
# THAM SO SINH DU LIEU
# ---------------------------------------------------------------------------
SEED          = 20261116
TOTAL_SAMPLES = 7000
VAL_RATIO     = 0.10
AUGMENT_COUNT = 1500
HORIZ_COUNT   = 5200   # so o ngang (128x64)
VERT_COUNT    = 1800   # so o doc  ( 64x128)

SYSTEM_PROMPT = "You are an OCR assistant."
# Prompt nay phai khop voi SHORT_PROMPT trong config.py (dung luc train va inference)
USER_PROMPT   = "The text in this image is a handwritten date in month/day format. Transcribe it. Output only the text."

random.seed(SEED)
np.random.seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NAP BACKGROUND
# ---------------------------------------------------------------------------
print("[1/5] Nap background tu background_MD...")
bg_files = sorted(list(BG_DIR.glob("*.png")))
if not bg_files:
    raise FileNotFoundError(
        f"Khong tim thay anh background trong: {BG_DIR}\n"
        f"Hay tai assets tu [DRIVE_LINK_PLACEHOLDER] va giai nen vao {BASE_DIR}"
    )


def load_clean_bg(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        b, g, r, a = cv2.split(im)
        alpha = a.astype(float) / 255.0
        b = (b * alpha + 255 * (1 - alpha)).astype(np.uint8)
        g = (g * alpha + 255 * (1 - alpha)).astype(np.uint8)
        r = (r * alpha + 255 * (1 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    dark = gray < 160
    if np.any(dark):
        gray[dark] = np.clip(gray[dark] * 0.35 + 105, 0, 245).astype(np.uint8)
    return gray


bg_mats = [load_clean_bg(p) for p in bg_files]
print(f"  Da nap {len(bg_mats)} background.")


# ---------------------------------------------------------------------------
# 2. KHO KY TU (ETL-3 + User drawn + Crops thuc te)
# ---------------------------------------------------------------------------
print("[2/5] Nap va can bang kho ky tu...")

drawings_path = USER_DIR / "drawings.jsonl"
drawings = []
if drawings_path.exists():
    with open(drawings_path, "r", encoding="utf-8") as f:
        drawings = [json.loads(line) for line in f if line.strip()]

d7_user    = [BASE_DIR / d["image"] for d in drawings if d.get("label") == "7"]
d8_user    = [BASE_DIR / d["image"] for d in drawings if d.get("label") == "8"]
d5_user    = [BASE_DIR / d["image"] for d in drawings if d.get("label") == "5"]
d5_crops   = sorted(list(CROPS_5_DIR.glob("*.png"))) if CROPS_5_DIR.exists() else []
d5_all_new = d5_user + d5_crops

digit_pool: dict = defaultdict(list)

etl7 = sorted(list((ETL_DIR / "7").glob("*.png")))[:200 - len(d7_user)]
digit_pool[7] = etl7 + d7_user

etl8 = sorted(list((ETL_DIR / "8").glob("*.png")))[:200 - len(d8_user)]
digit_pool[8] = etl8 + d8_user

etl5 = sorted(list((ETL_DIR / "5").glob("*.png")))[:200 - len(d5_all_new)]
digit_pool[5] = etl5 + d5_all_new

for d in [0, 1, 2, 3, 4, 6, 9]:
    folder = ETL_DIR / str(d)
    digit_pool[d] = sorted(list(folder.glob("*.png")))[:200]

d1_pool = sorted(list(D1_DIR.glob("*.png"))) if D1_DIR.exists() else []

print("  Thong ke kho ky tu:")
for d in range(10):
    p_len = len(digit_pool[d])
    n_user = (len(d7_user) if d == 7 else
              len(d8_user) if d == 8 else
              len(d5_all_new) if d == 5 else 0)
    print(f"    - So {d}: {p_len} mau (bo sung moi: {n_user})")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def crop_tight(gray: np.ndarray, thresh: int = 185, pad: int = 1):
    mask = gray < thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return gray[
        max(0, rmin - pad):min(gray.shape[0] - 1, rmax + pad) + 1,
        max(0, cmin - pad):min(gray.shape[1] - 1, cmax + pad) + 1,
    ]


def get_digit_img(d: int) -> np.ndarray:
    if d == 1 and len(d1_pool) > 0 and random.random() < 0.35:
        f = random.choice(d1_pool)
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thinned = cv2.dilate(im, k, iterations=1)
        blended = cv2.addWeighted(im, 0.35, thinned, 0.65, 0)
        blurred = cv2.GaussianBlur(blended, (3, 3), 0.5)
        cr = crop_tight(blurred, thresh=200, pad=1)
        if cr is not None and cr.shape[0] >= 10:
            return cr

    f = random.choice(digit_pool[d])
    im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    cr = crop_tight(im, thresh=185, pad=1)
    if cr is None or cr.shape[0] < 10:
        cr = im

    r = random.random()
    if r < 0.15:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cr = cv2.addWeighted(cr, 0.5, cv2.dilate(cr, k, iterations=1), 0.5, 0)
    elif r < 0.30:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cr = cv2.addWeighted(cr, 0.6, cv2.erode(cr, k, iterations=1), 0.4, 0)
    return cr


def paste_token(canvas: np.ndarray, token: np.ndarray, x: int, y: int, ink_val: int = 25):
    th, tw = token.shape[:2]
    ch, cw = canvas.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(cw, x + tw); y1 = min(ch, y + th)
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    tok_crop = token[y0 - y:y1 - y, x0 - x:x1 - x]
    mask = tok_crop < 190
    if not np.any(mask):
        return
    factor = np.clip((tok_crop[mask].astype(float) - 20) / 170.0, 0, 1)
    target_ink = ink_val + factor * 35
    roi[mask] = np.clip(np.minimum(roi[mask], target_ink), 0, 255).astype(np.uint8)
    canvas[y0:y1, x0:x1] = roi


# ---------------------------------------------------------------------------
# 3. SINH O NGANG VA O DOC
# ---------------------------------------------------------------------------
def generate_date_cell_horiz(label: str) -> np.ndarray:
    m_str, d_str = label.split("/")
    bg = random.choice(bg_mats).copy()
    h, w = bg.shape[:2]

    dark = bg < 160
    ys, xs = np.where(dark)
    slash_xmin, slash_xmax = xs.min(), xs.max()
    slash_ymin, slash_ymax = ys.min(), ys.max()
    slash_h = slash_ymax - slash_ymin + 1
    cy = int(ys.mean())

    dh = max(18, min(int(slash_h * random.uniform(0.68, 0.85)), h - 10))

    def build_num_img(s):
        tokens = []
        for c in s:
            im_d = get_digit_img(int(c))
            sc = dh / im_d.shape[0]
            nw = max(3, int(im_d.shape[1] * sc))
            tokens.append(cv2.resize(im_d, (nw, dh),
                                     interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR))
        if len(tokens) == 1:
            return tokens[0]
        gap = np.ones((dh, random.randint(1, 2)), np.uint8) * 255
        return np.hstack([tokens[0], gap, tokens[1]])

    m_img = build_num_img(m_str)
    d_img = build_num_img(d_str)
    mw, dw = m_img.shape[1], d_img.shape[1]

    gap_m = random.choice([0, 1, 2, 3, 4, 5])
    gap_d = random.choice([0, 1, 2, 3, 4, 5])
    x_m = max(2, slash_xmin - mw - gap_m)
    y_m = max(2, min(h - dh - 2, cy - dh // 2 + random.randint(-2, 2)))
    x_d = min(w - dw - 2, slash_xmax + gap_d)
    y_d = max(2, min(h - dh - 2, cy - dh // 2 + random.randint(-2, 2)))

    ink = random.randint(18, 38)
    paste_token(bg, m_img, x_m, y_m, ink_val=ink)
    paste_token(bg, d_img, x_d, y_d, ink_val=ink)

    if abs(w / h - 2.0) < 0.25:
        return cv2.resize(bg, (128, 64), interpolation=cv2.INTER_AREA)
    target_h = int(w * 0.5)
    if target_h <= h:
        y0 = max(0, min(h - target_h, cy - target_h // 2))
        crop = bg[y0:y0 + target_h, 0:w]
    else:
        crop = bg
    return cv2.resize(crop, (128, 64), interpolation=cv2.INTER_AREA)


def generate_date_cell_vert(label: str) -> np.ndarray:
    m_str, d_str = label.split("/")
    vert_candidates = [bg_mats[i] for i in [0, 3, 4, 5] if i < len(bg_mats)]
    if not vert_candidates:
        vert_candidates = bg_mats
    bg = random.choice(vert_candidates).copy()
    h, w = bg.shape[:2]

    dark = bg < 160
    ys, xs = np.where(dark)
    slash_ymin, slash_ymax = ys.min(), ys.max()
    cx = int(xs.mean())

    top_space = slash_ymin
    bot_space = h - slash_ymax
    dh = max(18, min(int(min(top_space, bot_space) * 0.75), 36))

    def build_num_img(s):
        tokens = []
        for c in s:
            im_d = get_digit_img(int(c))
            sc = dh / im_d.shape[0]
            nw = max(3, int(im_d.shape[1] * sc))
            tokens.append(cv2.resize(im_d, (nw, dh),
                                     interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR))
        if len(tokens) == 1:
            return tokens[0]
        gap = np.ones((dh, random.randint(1, 2)), np.uint8) * 255
        return np.hstack([tokens[0], gap, tokens[1]])

    m_img = build_num_img(m_str)
    d_img = build_num_img(d_str)
    mw, dw = m_img.shape[1], d_img.shape[1]

    gap_top = random.choice([0, 1, 2, 3, 4])
    gap_bot = random.choice([0, 1, 2, 3, 4])
    y_m = max(2, slash_ymin - dh - gap_top)
    x_m = max(2, min(w - mw - 2, cx - mw // 2 + random.randint(-2, 2)))
    y_d = min(h - dh - 2, slash_ymax + gap_bot)
    x_d = max(2, min(w - dw - 2, cx - dw // 2 + random.randint(-2, 2)))

    ink = random.randint(18, 38)
    paste_token(bg, m_img, x_m, y_m, ink_val=ink)
    paste_token(bg, d_img, x_d, y_d, ink_val=ink)

    target_w = int(h * 0.5)
    if target_w <= w:
        x0 = max(0, min(w - target_w, cx - target_w // 2))
        crop = bg[0:h, x0:x0 + target_w]
    else:
        crop = bg
    return cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)


def apply_blur_and_tilt(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    angle = random.uniform(-6.0, 6.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    sigma = random.uniform(0.5, 0.9)
    return cv2.GaussianBlur(rotated, (3, 3), sigma)


# ---------------------------------------------------------------------------
# 4. LAP KE HOACH - PHAN BO DEU TẤT CA NGAY TRONG LICH
# ---------------------------------------------------------------------------
print("[3/5] Lap danh sach nhan theo phan bo deu...")

days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}
all_calendar_dates = [
    f"{m}/{d}" for m in range(1, 13) for d in range(1, days_in_month[m] + 1)
]
n_dates = len(all_calendar_dates)   # 365

# Moi ngay duoc lap di lap lai de dat TOTAL_SAMPLES, doi du thi lap vong
labels_list = []
reps_full = TOTAL_SAMPLES // n_dates         # so lan lap nguyen cua toan bo lich
remainder  = TOTAL_SAMPLES % n_dates         # so ngay them

for _ in range(reps_full):
    labels_list.extend(all_calendar_dates)

if remainder > 0:
    labels_list.extend(all_calendar_dates[:remainder])

assert len(labels_list) == TOTAL_SAMPLES

random.shuffle(labels_list)

aug_indices  = set(random.sample(range(TOTAL_SAMPLES), AUGMENT_COUNT))
vert_indices = set(random.sample(range(TOTAL_SAMPLES), VERT_COUNT))

print(f"  Tong mau: {TOTAL_SAMPLES:,} | {n_dates} ngay x {reps_full} lan + {remainder} du")
print(f"  Phan bo: {TOTAL_SAMPLES - VERT_COUNT} o ngang (128x64) + {VERT_COUNT} o doc (64x128)")
print(f"  Augment nhe (tilt + blur): {AUGMENT_COUNT} anh")


# ---------------------------------------------------------------------------
# 5. SINH DU LIEU
# ---------------------------------------------------------------------------
print("[4/5] Dang sinh du lieu...")
all_items = []
t0 = time.time()

for idx, label in enumerate(labels_list):
    img_rel  = f"train/crop_{idx:05d}.png"
    img_path = OUT_DIR / img_rel

    is_vert = idx in vert_indices
    mat = generate_date_cell_vert(label) if is_vert else generate_date_cell_horiz(label)

    is_aug = idx in aug_indices
    if is_aug:
        mat = apply_blur_and_tilt(mat)

    cv2.imwrite(str(img_path), mat)

    item = {
        "image": img_rel,
        "label": label,
        "orientation": "vertical" if is_vert else "horizontal",
        "augmented": is_aug,
        "messages": [
            {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",      "content": [{"type": "image", "image": img_rel},
                                              {"type": "text",  "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text",  "text": label}]},
        ],
    }
    all_items.append(item)

    if (idx + 1) % 1000 == 0 or idx == TOTAL_SAMPLES - 1:
        elapsed = time.time() - t0
        print(f"  Da sinh {idx + 1:,} / {TOTAL_SAMPLES:,} anh ({elapsed:.1f}s)...")


# ---------------------------------------------------------------------------
# CHIA TRAIN / VAL (90% / 10%)
# ---------------------------------------------------------------------------
val_count   = int(TOTAL_SAMPLES * VAL_RATIO)
train_items = all_items[val_count:]
val_items   = all_items[:val_count]

with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
    for item in train_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(VAL_JSONL, "w", encoding="utf-8") as f:
    for item in val_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

stats = {
    "total_generated":    TOTAL_SAMPLES,
    "train_count":        len(train_items),
    "val_count":          len(val_items),
    "horizontal_count":   TOTAL_SAMPLES - VERT_COUNT,
    "vertical_count":     VERT_COUNT,
    "augmented_count":    AUGMENT_COUNT,
    "distribution":       "uniform (all 365 calendar dates equally)",
    "user_supplemented_digits": {
        "7": len(d7_user),
        "8": len(d8_user),
        "5": len(d5_all_new),
    },
    "prompt":     USER_PROMPT,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(STATS_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n[5/5] Hoan tat tai: {OUT_DIR}")
print(f"  Train : {len(train_items):,} mau  -->  {TRAIN_JSONL}")
print(f"  Val   : {len(val_items):,} mau  -->  {VAL_JSONL}")
print(f"  Stats : {STATS_JSON}")
