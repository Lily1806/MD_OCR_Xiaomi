MD_OCR_Xiaomi - Fine-tuning LoRA cho nhan dang ngay thang M/D tren Xiaomi-OCR-0

Repo nay chua code sinh du lieu va huan luyen LoRA adapter nhan dang ngay thang dinh dang thang/ngay (M/D) viet tay tren bien the Xiaomi-OCR-0.

Ket qua tot nhat dat duoc: 711/720 o (98.75%) tren tap test thuc te.


Cau truc repo
-------------

    MD_OCR_Xiaomi/
        scripts/
            config.py          Tat ca tham so: duong dan, sieu tham so, prompt
            gen_dataset.py     Sinh du lieu M/D (phan bo deu 365 ngay)
            train.py           Huan luyen LoRA (chay local co GPU hoac CPU)
            eval.py            Danh gia adapter tren tap val hoac tap test
            merge_lora.py      Tron nhieu adapter bang weighted average
        data/
            README_assets.md   Huong dan tai va cai dat assets
        docs/
            GUIDE_VI.md        Huong dan day du bang tieng Viet
        README.md              File nay


Yeu cau he thong
----------------

- Python >= 3.9
- torch >= 2.0 (co CUDA neu co GPU)
- transformers >= 5.17.0
- peft >= 0.10
- accelerate >= 0.28
- safetensors
- opencv-python (cv2)
- pillow
- einops, sentencepiece

Cai dat tat ca:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install transformers==5.17.0 peft>=0.10 accelerate>=0.28 safetensors
    pip install opencv-python pillow einops sentencepiece huggingface_hub


Bat dau nhanh
-------------

Buoc 1: Tai assets
    Tai file ZIP tu Google Drive: [DRIVE_LINK_PLACEHOLDER]
    Giai nen vao thu muc cua ban (vi du: /data/md_ocr_assets)
    Xem chi tiet: data/README_assets.md

Buoc 2: Sinh du lieu
    export MD_OCR_BASE=/data/md_ocr_assets
    python3 scripts/gen_dataset.py
    Ket qua: /data/md_ocr_assets/dataset_date_md/ (7000 anh, train.jsonl, val.jsonl)

Buoc 3: Chinh config
    Mo scripts/config.py va chinh:
    - DATASET_DIR  : duong dan toi du lieu vua sinh
    - OUTPUT_DIR   : noi luu adapter
    - WARM_START_ADAPTER : None (train tu dau) hoac duong dan adapter co san
    - NUM_EPOCHS, LR, ... : sieu tham so

Buoc 4: Huan luyen
    MD_OCR_BASE=/data/md_ocr_assets python3 scripts/train.py

Buoc 5: Danh gia
    # Tren tap val:
    MD_OCR_BASE=/data/md_ocr_assets python3 scripts/eval.py

    # Tren 720 o test thuc te (neu co):
    MD_OCR_BASE=/data/md_ocr_assets python3 scripts/eval.py --test720 /duong/dan/eval_720


Gop adapter (Merge LoRA)
------------------------

Tron 70% adapter_v9 va 30% adapter_v15:
    python3 scripts/merge_lora.py \
        --adapters /path/adapter_v9 /path/adapter_v15 \
        --weights  0.7 0.3 \
        --output   /path/merged_7030

Tron 3 adapter:
    python3 scripts/merge_lora.py \
        --adapters a1 a2 a3 \
        --weights  0.5 0.3 0.2 \
        --output   merged_3way


Prompt dung cho mo hinh
-----------------------

Data: "The text in this image is a handwritten date in month/day format. Transcribe it. Output only the text."
Train: Giong het data (xem SHORT_PROMPT trong config.py)
Inference: Giong het SHORT_PROMPT


Ket qua benchmark
-----------------

Mo hinh                     So o dung   Ty le
Merged LoRA 70/30 (V9+V15)  711 / 720   98.75%   (tot nhat)
V9 (Base Warmstart)         709 / 720   98.47%
V15 (Train moi 15k anh)     708 / 720   98.33%


License
-------
MIT
