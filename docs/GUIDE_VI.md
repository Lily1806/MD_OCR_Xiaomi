Huong dan su dung MD_OCR_Xiaomi
================================
Huan luyen LoRA nhan dang ngay thang dinh dang thang/ngay (M/D) viet tay
tren bien the mo hinh Xiaomi-OCR-0.


1. CHUAN BI MOI TRUONG
----------------------

Cai dat cac thu vien can thiet:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install transformers==5.17.0 peft>=0.10 accelerate>=0.28 safetensors
    pip install opencv-python pillow einops sentencepiece huggingface_hub

Neu khong co GPU CUDA, dung phien ban CPU cua PyTorch:
    pip install torch torchvision
    (Luu y: huan luyen tren CPU rat cham, chi dung de thu nghiem.)

Dang nhap Hugging Face de tai mo hinh goc:
    huggingface-cli login
    (Nhap token tai https://huggingface.co/settings/tokens)


2. TAI ASSETS
-------------

Tai file ZIP chua du lieu ky tu viet tay tu Google Drive:
    Link: [DRIVE_LINK_PLACEHOLDER]

Giai nen vao thu muc cua ban, vi du /data/md_ocr_assets:
    unzip md_ocr_assets.zip -d /data/md_ocr_assets

Cau truc thu muc sau khi giai nen:
    /data/md_ocr_assets/
        ETL_Project/output/characters/0/ ... 9/
        handwritten_gen_package/background_MD/
        handwritten_gen_package/handwritten_digits_2000/1/
        user_drawn_samples/drawings.jsonl + images/
        raw_handwritten_crops/crops_five_new/


3. SINH DU LIEU
---------------

Dat bien moi truong MD_OCR_BASE tro den thu muc chua assets:
    export MD_OCR_BASE=/data/md_ocr_assets

Chay script sinh du lieu:
    python3 scripts/gen_dataset.py

Qua trinh nay sinh khoang 7000 anh vao thu muc:
    /data/md_ocr_assets/dataset_date_md/
        train.jsonl    (6300 mau)
        val.jsonl      (700 mau)
        train/         (anh PNG)
        stats.json     (thong ke)

Du lieu phan bo deu tren 365 ngay trong nam (moi ngay xap xi 19 mau).
Khong co ngay nao duoc uu tien dac biet.


4. CHINH THAM SO HUAN LUYEN
----------------------------

Mo file scripts/config.py bang trinh soan thao.
Cac tham so quan trong:

    # --- Duong dan ---
    BASE_DIR    = Path(os.environ.get("MD_OCR_BASE", "."))
    DATASET_DIR = BASE_DIR / "dataset_date_md"
    OUTPUT_DIR  = BASE_DIR / "output_lora"

    # --- Warm-start ---
    # Dat None de train tu dau (fresh start).
    # Dat duong dan adapter de khoi dong tu checkpoint co san.
    WARM_START_ADAPTER = None
    # Vi du: WARM_START_ADAPTER = BASE_DIR / "lora_adapter_v9"

    # --- Sieu tham so ---
    NUM_EPOCHS  = 5
    LR          = 3e-5
    LORA_R      = 32
    LORA_ALPHA  = 32

Khong can sua file train.py, chi sua config.py.


5. HUAN LUYEN
-------------

Chay lenh sau (dat bien MD_OCR_BASE neu chua dat o tren):

    MD_OCR_BASE=/data/md_ocr_assets python3 scripts/train.py

Qua trinh huan luyen se:
    - Hien thi log moi 10 step (loss, eval_loss)
    - Luu checkpoint sau moi 100 step
    - Luu adapter tot nhat vao: /data/md_ocr_assets/output_lora/lora_adapter/

Neu bi ngat giua chung, chay lai se tu dong tiep tuc tu checkpoint cuoi.

Thoi gian uoc tinh:
    - GPU P100 (Kaggle): khoang 30-60 phut tuy so luong mau va epoch
    - GPU cao cap hon (A100): 10-20 phut
    - CPU: nhieu gio


6. DANH GIA
-----------

Danh gia tren tap val (du lieu sinh):
    MD_OCR_BASE=/data/md_ocr_assets python3 scripts/eval.py

Danh gia tren tap test 720 o thuc te (neu co):
    MD_OCR_BASE=/data/md_ocr_assets python3 scripts/eval.py \
        --test720 /duong/dan/eval_720 \
        --out ket_qua.json

Gioi han so mau:
    python3 scripts/eval.py --n 100


7. GOP ADAPTER (MERGE LORA)
----------------------------

Tron 2 adapter theo ty le 70/30:
    python3 scripts/merge_lora.py \
        --adapters /path/adapter_v9 /path/adapter_v15 \
        --weights  0.7 0.3 \
        --output   /path/merged_7030

Tron 3 adapter:
    python3 scripts/merge_lora.py \
        --adapters a1 a2 a3 \
        --weights  0.5 0.3 0.2 \
        --output   merged_3way


8. PUSH CODE LEN GITHUB
------------------------

Buoc 1: Khoi tao git trong thu muc repo (chi can lam lan dau tien)

    cd /Users/tranly/TuHoc_AI/Cell/MD_OCR_Xiaomi_repo
    git init
    git remote add origin git@github.com:Lily1806/MD_OCR_Xiaomi.git

Buoc 2: Kiem tra ket noi SSH

    ssh -T git@github.com
    # Ket qua dung: "Hi Lily1806! You've successfully authenticated..."
    # Neu loi: kiem tra SSH key tai https://github.com/settings/keys

Buoc 3: Them file va commit

    git add .
    git status    # kiem tra cac file se push (dam bao khong co file lon)
    git commit -m "Initial commit: gen_dataset, train, eval, merge_lora scripts"

Buoc 4: Push len GitHub

    git push -u origin main
    # Neu branch la master thay vi main:
    # git push -u origin master


9. MO HINH VA PROMPT
---------------------

Mo hinh goc: xiaomi-research/Xiaomi-OCR-0 (Hugging Face)

Prompt dung de sinh du lieu va huan luyen:
    System: "You are an OCR assistant."
    User  : "The text in this image is a handwritten date in month/day format.
             Transcribe it. Output only the text."

Prompt phai giong nhau luc sinh du lieu (gen_dataset.py)
va luc huan luyen / suy luan (SHORT_PROMPT trong config.py).


10. CAU HOI THUONG GAP
-----------------------

Loi "Khong tim thay GPU":
    - Kiem tra CUDA: python3 -c "import torch; print(torch.cuda.is_available())"
    - Cai lai PyTorch voi CUDA dung phien ban

Loi "Loss NaN/Inf":
    - Giam LR trong config.py (thu LR = 1e-5)
    - Neu dung fp16 (khong co bf16): giam them, thu LR_FP16_CAP

Loi "Khong tim thay adapter":
    - Chay train.py truoc khi chay eval.py
    - Kiem tra ADAPTER_DIR trong config.py

Loi "Khong tim thay anh background":
    - Tai assets tu link Drive o Buoc 2 va giai nen dung cau truc
    - Kiem tra bien MD_OCR_BASE tro dung thu muc
