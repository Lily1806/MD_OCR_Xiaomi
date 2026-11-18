#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
------------------------------------------------------------------------------------------
Huan luyen LoRA cho mo hinh Xiaomi-OCR-0 nhan dang ngay thang M/D.
Chay tren GPU local (CUDA) hoac CPU (cham hon).

Yeu cau:
    pip install torch transformers>=5.17.0 peft>=0.10 accelerate>=0.28
                safetensors einops sentencepiece pillow

Cach chay:
    # Huan luyen tu dau (fresh start):
    python3 train.py

    # Warm-start tu adapter san co:
    # Dat WARM_START_ADAPTER trong config.py truoc, roi chay:
    python3 train.py

    # Ghi de bien moi truong:
    MD_OCR_BASE=/path/to/data python3 train.py
------------------------------------------------------------------------------------------
"""

import os
import sys
import json
import math
import time
import shutil
from pathlib import Path

# Nap config - import truc tiep tu config.py cung thu muc
sys.path.insert(0, str(Path(__file__).parent))
import config as CFG

# ---------------------------------------------------------------------------
# CHUAN BI MOI TRUONG
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.trainer_utils import get_last_checkpoint


# ---------------------------------------------------------------------------
# PATCH PEFT / TORCHAO CONFLICT
# ---------------------------------------------------------------------------
def patch_peft_torchao():
    for _n, _m in list(sys.modules.items()):
        if _n.startswith("peft") and hasattr(_m, "is_torchao_available"):
            _m.is_torchao_available = lambda: False


patch_peft_torchao()


# ---------------------------------------------------------------------------
# DETECT DEVICE
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE     = "cuda"
    USE_BF16   = torch.cuda.get_device_capability(0)[0] >= 8
    DTYPE      = torch.bfloat16 if USE_BF16 else torch.float16
    print(f"Su dung GPU: {torch.cuda.get_device_name(0)}")
    print(f"Precision  : {'bf16' if USE_BF16 else 'fp16'}")
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32
    print("Khong co GPU, chay tren CPU (rat cham, chi nen dung de thu nghiem).")

ADAPTER_DIR = CFG.ADAPTER_DIR
OUTPUT_DIR  = CFG.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset   : {CFG.DATASET_DIR}")
print(f"Output    : {OUTPUT_DIR}")
print(f"Warm-start: {CFG.WARM_START_ADAPTER or 'None (train tu dau)'}")


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _content_text(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def get_label(rec):
    lab = rec.get("label")
    if lab is not None:
        return lab if isinstance(lab, str) else json.dumps(lab, ensure_ascii=False)
    return _content_text(rec["messages"][-1]["content"])


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


class OCRDataset(Dataset):
    def __init__(self, jsonl_path: Path, root: Path, prompt_override: str = None):
        self.root = root
        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f if l.strip()]
        if prompt_override:
            for rec in self.records:
                for turn in rec["messages"]:
                    if turn["role"] == "user" and isinstance(turn["content"], list):
                        for part in turn["content"]:
                            if part.get("type") == "text":
                                part["text"] = prompt_override

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.root / rec["image"]
        image = Image.open(img_path).convert("RGB")
        return {"image": image, "messages": rec["messages"], "label": get_label(rec)}


class OCRDataCollator:
    def __init__(self, processor, max_length: int = 512):
        self.processor  = processor
        self.max_length = max_length

        def _id(tok):
            ids = processor.tokenizer.encode(tok, add_special_tokens=False)
            return ids[-1] if ids else None

        self.im_end_id    = _id("<|im_end|>")
        self.im_start_id  = _id("<|im_start|>")
        self.think_end_id = _id("</think>")
        self._warned = False
        self.n_empty = 0

    def _find_span(self, l):
        ends = [j for j, t in enumerate(l) if t == self.im_end_id] if self.im_end_id else []
        if not ends:
            return None
        start = None
        if self.think_end_id:
            think = [j for j, t in enumerate(l) if t == self.think_end_id]
            if think:
                start = think[-1] + 1
        if start is None and self.im_start_id:
            st = [j for j, t in enumerate(l) if t == self.im_start_id]
            if st:
                start = st[-1] + 1
                while start < len(l) and l[start] != self.im_end_id and \
                        self.processor.tokenizer.decode([l[start]]).strip() in ("", "assistant"):
                    start += 1
        if start is None:
            return None
        while start < len(l) and l[start] != self.im_end_id and \
                self.processor.tokenizer.decode([l[start]]).strip() == "":
            start += 1
        end = ends[-1]
        return (start, end) if start <= end else None

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        texts  = [
            self.processor.apply_chat_template(
                clean_messages(b["messages"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            for b in batch
        ]
        enc = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        if enc["input_ids"].shape[1] > self.max_length and not self._warned:
            print(f"Canh bao: chuoi dai {enc['input_ids'].shape[1]} > {self.max_length} token")
            self._warned = True
        ids_all = enc["input_ids"]
        labels  = torch.full_like(ids_all, -100)
        for i, ids in enumerate(ids_all):
            span = self._find_span(ids.tolist())
            if span is None:
                self.n_empty += 1
                continue
            s, e = span
            labels[i, s : e + 1] = ids[s : e + 1]
        enc["labels"] = labels
        return enc


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------
class TimeLimit(TrainerCallback):
    def __init__(self, hours):
        self.limit = hours * 3600
        self.t0    = None

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        if self.t0 and time.time() - self.t0 > self.limit:
            print("Het gio -> luu adapter va dung.")
            control.should_training_stop = True
        return control


class Heartbeat(TrainerCallback):
    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        msg = f"[{time.time()-self.t0:.0f}s] step {state.global_step}/{state.max_steps}"
        if logs:
            if "loss" in logs:
                msg += f"  loss={logs['loss']:.4f}"
            if "eval_loss" in logs:
                msg += f"  eval_loss={logs['eval_loss']:.4f}"
        print(msg, flush=True)


class NaNGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            for k in ("loss", "eval_loss"):
                if k in logs and (math.isnan(logs[k]) or math.isinf(logs[k])):
                    raise RuntimeError(
                        f"Loss bi NaN/Inf tai step {state.global_step}. "
                        "Giam LR trong config.py hoac kiem tra du lieu."
                    )


# ---------------------------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------------------------
def build_model(processor):
    print("Dang tai mo hinh goc...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            CFG.MODEL_ID,
            dtype=DTYPE,
            device_map={"": DEVICE} if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            CFG.MODEL_ID,
            torch_dtype=DTYPE,
            device_map={"": DEVICE} if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
    model.config.use_cache = False

    LANG_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets, n_lang, n_vis = [], 0, 0
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        is_vis = "visual" in n or "vision" in n
        if is_vis:
            if CFG.INCLUDE_VISION:
                targets.append(n); n_vis += 1
        elif n.split(".")[-1] in LANG_KEYS:
            targets.append(n); n_lang += 1
    print(f"LoRA: {n_lang} lang + {n_vis} vision  (r={CFG.LORA_R}, alpha={CFG.LORA_ALPHA})")
    assert targets, "Khong tim thay lop Linear nao. Kiem tra lai MODEL_ID."

    warm = CFG.WARM_START_ADAPTER
    if warm and Path(warm).exists():
        print(f"Warm-start: Nap adapter tu {warm}")
        model = PeftModel.from_pretrained(model, str(warm), is_trainable=True)
    else:
        if warm:
            print(f"Canh bao: WARM_START_ADAPTER khong tim thay: {warm} -> train tu dau.")
        print("Khoi tao LoRA moi tu dau.")
        model = get_peft_model(
            model,
            LoraConfig(
                r=CFG.LORA_R,
                lora_alpha=CFG.LORA_ALPHA,
                target_modules=targets,
                lora_dropout=CFG.LORA_DROPOUT,
                bias="none",
            ),
        )

    # Chuyen sang float32 de tinh on dinh hon khi train
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # Nap processor
    print("Dang nap processor...")
    processor = AutoProcessor.from_pretrained(CFG.MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Dataset
    dataset_dir = CFG.DATASET_DIR
    train_jsonl = dataset_dir / "train.jsonl"
    val_jsonl   = dataset_dir / "val.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Khong tim thay {train_jsonl}. Chay gen_dataset.py truoc."
        )

    print("Dang nap dataset...")
    train_ds = OCRDataset(train_jsonl, dataset_dir, prompt_override=CFG.SHORT_PROMPT)
    val_ds   = OCRDataset(val_jsonl,   dataset_dir, prompt_override=CFG.SHORT_PROMPT)
    collator = OCRDataCollator(processor)
    assert collator.im_end_id is not None, (
        "Tokenizer khong co <|im_end|>. Kiem tra lai mo hinh."
    )
    print(f"Train: {len(train_ds):,} mau | Val: {len(val_ds):,} mau")

    # Model
    model = build_model(processor)

    # Tinh toan gradient accumulation
    grad_accum = max(1, math.ceil(CFG.TARGET_EFF_BS / CFG.PER_DEVICE_BS))
    eff_bs     = CFG.PER_DEVICE_BS * grad_accum
    print(f"Batch: per_device={CFG.PER_DEVICE_BS}, accum={grad_accum}, eff={eff_bs}")

    # Precision flags
    use_fp16 = (DEVICE == "cuda") and not USE_BF16
    use_bf16 = (DEVICE == "cuda") and USE_BF16

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=CFG.NUM_EPOCHS,
        per_device_train_batch_size=CFG.PER_DEVICE_BS,
        per_device_eval_batch_size=CFG.PER_DEVICE_BS,
        gradient_accumulation_steps=grad_accum,
        learning_rate=CFG.LR,
        lr_scheduler_type=CFG.LR_SCHEDULER,
        warmup_steps=CFG.WARMUP_STEPS,
        weight_decay=CFG.WEIGHT_DECAY,
        logging_steps=CFG.LOG_STEPS,
        eval_strategy="steps" if CFG.EVAL_STEPS else "epoch",
        eval_steps=CFG.EVAL_STEPS,
        save_strategy="steps" if CFG.SAVE_STEPS else "epoch",
        save_steps=CFG.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        report_to="none",
        seed=CFG.SEED,
        remove_unused_columns=False,
    )

    callbacks = [Heartbeat(), NaNGuard()]
    if CFG.TIME_LIMIT_H:
        callbacks.append(TimeLimit(CFG.TIME_LIMIT_H))

    # Resume tu checkpoint neu co
    last_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
    if last_ckpt:
        print(f"Tim thay checkpoint: {last_ckpt}. Tiep tuc tu day.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=Subset(val_ds, list(range(min(400, len(val_ds))))),
        data_collator=collator,
        callbacks=callbacks,
    )

    # Bat dau train
    print(f"\nBat dau huan luyen | {CFG.NUM_EPOCHS} epoch | LR={CFG.LR}\n")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Luu adapter
    print("\nLuu adapter...")
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))

    # Xoa checkpoint trung gian
    for d in OUTPUT_DIR.glob("checkpoint-*"):
        shutil.rmtree(d, ignore_errors=True)

    # Thong bao ket qua
    total_min = (time.time() - t_start) / 60
    print(f"\nHoan tat huan luyen! Tong thoi gian: {total_min:.1f} phut")
    print(f"Adapter da luu tai: {ADAPTER_DIR}")
    print("Cac file trong adapter:")
    for f in sorted(ADAPTER_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

