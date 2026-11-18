#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py
------------------------------------------------------------------------------------------
Danh gia LoRA adapter tren tap val hoac tap test 720 o M/D thuc te.

Yeu cau: torch, transformers, peft, pillow
Cach chay:
    # Danh gia tren tap val (lay tu dataset_date_md/val.jsonl)
    python3 eval.py

    # Danh gia tren tap test 720 o (neu co eval_720/eval_720.jsonl)
    python3 eval.py --test720 /duong/dan/toi/eval_720

    # Gioi han so luong mau
    python3 eval.py --n 100
------------------------------------------------------------------------------------------
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as CFG

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


def patch_peft_torchao():
    for _n, _m in list(sys.modules.items()):
        if _n.startswith("peft") and hasattr(_m, "is_torchao_available"):
            _m.is_torchao_available = lambda: False


patch_peft_torchao()


def cer(gt: str, pred: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    m, n = len(gt), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if gt[i - 1] == pred[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev  = temp
    return dp[n] / m


def clean_messages(msgs):
    out = []
    for turn in msgs:
        if turn["role"] == "user" and isinstance(turn["content"], list):
            content = [
                {"type": "image"} if p.get("type") == "image" else p
                for p in turn["content"]
            ]
            out.append({"role": turn["role"], "content": content})
        else:
            out.append(turn)
    return out


def load_model_and_processor():
    if torch.cuda.is_available():
        USE_BF16 = torch.cuda.get_device_capability(0)[0] >= 8
        DTYPE    = torch.bfloat16 if USE_BF16 else torch.float16
        dev_map  = {"": 0}
    else:
        DTYPE   = torch.float32
        dev_map = None

    processor = AutoProcessor.from_pretrained(CFG.MODEL_ID, trust_remote_code=True)

    adapter_dir = CFG.ADAPTER_DIR
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Khong tim thay adapter tai: {adapter_dir}\n"
            f"Chay train.py truoc, hoac dat dung ADAPTER_DIR trong config.py."
        )
    print(f"Nap adapter tu: {adapter_dir}")

    try:
        base = AutoModelForImageTextToText.from_pretrained(
            CFG.MODEL_ID, dtype=DTYPE, device_map=dev_map, trust_remote_code=True
        )
    except TypeError:
        base = AutoModelForImageTextToText.from_pretrained(
            CFG.MODEL_ID, torch_dtype=DTYPE, device_map=dev_map, trust_remote_code=True
        )

    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    return model, processor, DTYPE if dev_map else torch.float32, dev_map


def predict_one(rec, model, processor, dtype, dev_map, root: Path, max_new: int = 32):
    img_path = root / rec["image"]
    image = Image.open(img_path).convert("RGB")
    try:
        text = processor.apply_chat_template(
            clean_messages(rec["messages"][:-1]),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            clean_messages(rec["messages"][:-1]),
            tokenize=False, add_generation_prompt=True,
        )
    inp = processor(text=text, images=[image], return_tensors="pt")
    if dev_map:
        inp = {
            k: (v.to("cuda", dtype=dtype) if v.is_floating_point() else v.to("cuda"))
            for k, v in inp.items()
        }
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=True)
    return processor.tokenizer.decode(
        out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_eval(records, root: Path, model, processor, dtype, dev_map, n: int, label: str):
    n = min(n, len(records))
    n_ok = 0; total_cer = 0.0; rows = []
    for i, rec in enumerate(records[:n]):
        gt   = rec.get("gt") or rec["messages"][-1]["content"][-1]["text"]
        pred = predict_one(rec, model, processor, dtype, dev_map, root)
        ok   = pred.strip() == gt.strip()
        c    = cer(gt, pred)
        n_ok += ok; total_cer += c
        rows.append({"image": rec.get("image", ""), "gt": gt, "pred": pred,
                     "ok": ok, "cer": round(c, 3)})
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] exact={n_ok/(i+1)*100:.1f}%  CER={total_cer/(i+1)*100:.2f}%")

    print(f"\n--- Ket qua {label} ---")
    print(f"So mau danh gia : {n}")
    print(f"Exact match     : {n_ok}/{n} = {n_ok/n*100:.2f}%")
    print(f"Average CER     : {total_cer/n*100:.2f}%")
    wrong = [r for r in rows if not r["ok"]]
    print(f"So o sai        : {len(wrong)}")
    for r in wrong[:10]:
        print(f"  GT: {r['gt']!r:20s}  PRED: {r['pred']!r:20s}  img: {r['image']}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,  default=9999,
                        help="Gioi han so mau danh gia")
    parser.add_argument("--test720", type=str,  default=None,
                        help="Duong dan thu muc chua eval_720.jsonl + anh thuc te")
    parser.add_argument("--out",     type=str,  default=None,
                        help="Luu ket qua JSON ra file")
    args = parser.parse_args()

    model, processor, dtype, dev_map = load_model_and_processor()

    all_rows = []

    if args.test720:
        test_dir  = Path(args.test720)
        test_jsonl = test_dir / "eval_720.jsonl"
        with open(test_jsonl, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        rows = run_eval(recs, test_dir, model, processor, dtype, dev_map, args.n, "tap 720 o thuc te")
        all_rows.extend(rows)
    else:
        val_jsonl = CFG.DATASET_DIR / "val.jsonl"
        with open(val_jsonl, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        rows = run_eval(recs, CFG.DATASET_DIR, model, processor, dtype, dev_map, args.n, "tap val")
        all_rows.extend(rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=1)
        print(f"\nDa luu ket qua: {args.out}")


if __name__ == "__main__":
    main()

