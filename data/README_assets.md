Assets cho viec sinh du lieu khong duoc luu trong repo vi kich thuoc lon.

Tai file assets tai Google Drive:
[DRIVE_LINK_PLACEHOLDER]

Sau khi tai ve, giai nen vao thu muc goc cua du an (hoac dat trong bien moi truong MD_OCR_BASE).
Cau truc thu muc sau khi giai nen:

<thu_muc_goc>/
    ETL_Project/
        output/
            characters/
                0/   (chua anh PNG)
                1/
                ...
                9/
    handwritten_gen_package/
        background_MD/      (anh nen: 1.png - 6.png)
        handwritten_digits_2000/
            1/              (anh so 1 viet tay)
    user_drawn_samples/
        drawings.jsonl
        images/
    raw_handwritten_crops/
        crops_five_new/     (anh so 5 cat thuc te)

Sau khi giai nen, chay lenh sau de sinh du lieu:
    export MD_OCR_BASE=/duong/dan/toi/thu_muc_goc
    python3 scripts/gen_dataset.py
