import os, sys, json, math, time
sys.path.insert(0, os.environ.get('CC_WORK', '/kaggle/working'))
import ocr_common as cc
cc.prepare_env()

import torch
from pathlib import Path
from torch.utils.data import Subset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer,
                          TrainerCallback, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.trainer_utils import get_last_checkpoint
cc.patch_peft_torchao()

CFG        = json.load(open(os.environ['CFG_PATH']))
rank       = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world      = int(os.environ.get('WORLD_SIZE', 1))
is_main    = (rank == 0)
torch.cuda.set_device(local_rank)
USE_BF16   = torch.cuda.get_device_capability(local_rank)[0] >= 8
DTYPE      = torch.bfloat16 if USE_BF16 else torch.float16
T0         = time.time()


def log(*a):
    if is_main:
        print(*a, flush=True)


DATASET_DIR = Path(CFG['dataset_dir'])
OUTPUT_DIR  = Path(CFG['output_dir'])
log(f'world={world} | bs={CFG["per_device_bs"]} | accum={CFG["accum"]} | '
    f'eff={CFG["per_device_bs"]*CFG["accum"]*world} | {"bf16" if USE_BF16 else "fp16"} | lr={CFG["lr"]}')

processor = AutoProcessor.from_pretrained(CFG['model_id'], trust_remote_code=True)
processor.tokenizer.padding_side = 'right'
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

try:
    model = AutoModelForImageTextToText.from_pretrained(
        CFG['model_id'], dtype=DTYPE, device_map={'': local_rank}, trust_remote_code=True)
except TypeError:
    model = AutoModelForImageTextToText.from_pretrained(
        CFG['model_id'], torch_dtype=DTYPE, device_map={'': local_rank}, trust_remote_code=True)
model.config.use_cache = False

LANG_KEYS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
targets, n_lang, n_vis = [], 0, 0
for n, m in model.named_modules():
    if not isinstance(m, torch.nn.Linear):
        continue
    is_vis = ('visual' in n) or ('vision' in n)
    if is_vis:
        if CFG['include_vision']:
            targets.append(n); n_vis += 1
    elif n.split('.')[-1] in LANG_KEYS:
        targets.append(n); n_lang += 1
log(f'LoRA: {n_lang} lang + {n_vis} vision  (r={CFG["lora_r"]}, alpha={CFG["lora_alpha"]})')
assert targets, 'Khong tim thay lop Linear nao de gan LoRA'
warm_start_path = CFG.get('warm_start_adapter')
if warm_start_path and Path(warm_start_path).exists():
    log(f"==> [WARM-START] Nạp trọng số LoRA từ V1: {warm_start_path}")
    model = PeftModel.from_pretrained(model, str(warm_start_path), is_trainable=True)
else:
    log("==> Khởi tạo LoRA mới từ đầu")
    model = get_peft_model(model, LoraConfig(
        r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'],
        target_modules=targets,
        lora_dropout=CFG.get('lora_dropout', 0.05),
        bias='none'))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
if is_main:
    model.print_trainable_parameters()

prompt   = CFG.get('short_prompt')
train_ds = cc.OCRDataset(DATASET_DIR / 'train.jsonl', DATASET_DIR, prompt_override=prompt)
val_ds   = cc.OCRDataset(DATASET_DIR / 'val.jsonl',   DATASET_DIR, prompt_override=prompt)
eval_ds  = Subset(val_ds, list(range(min(400, len(val_ds)))))
SMOKE    = CFG.get('smoke', False)
if SMOKE:
    train_ds.records = train_ds.records[:96]
    eval_ds = Subset(val_ds, list(range(min(16, len(val_ds)))))
    log('SMOKE TEST: 96 mau, 6 step')
collator = cc.OCRDataCollator(processor)


class TimeLimit(TrainerCallback):
    def __init__(self, hours):
        self.limit = hours * 3600
        self.t0 = None
    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.limit:
            log('Het gio -> luu adapter va dung.')
            control.should_training_stop = True
        return control


class Heartbeat(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        msg = f'[{time.time()-T0:.0f}s] step {state.global_step}/{state.max_steps}'
        if logs:
            if 'loss' in logs:
                msg += f'  loss={logs["loss"]:.4f}'
            if 'eval_loss' in logs:
                msg += f'  eval_loss={logs["eval_loss"]:.4f}'
        log(msg)


class NaNGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            for k in ('loss', 'eval_loss'):
                if k in logs and not math.isfinite(float(logs[k])):
                    raise RuntimeError(f'{k} = NaN/Inf -> dung. Giam LR (vd 5e-5) hoac dung GPU ho tro bf16.')


bs, accum = CFG['per_device_bs'], CFG['accum']
steps_per_epoch = math.ceil(len(train_ds) / (bs * accum * world))
every = 3 if SMOKE else max(25, steps_per_epoch // 3)
log(f'steps_per_epoch={steps_per_epoch} | eval+save every={every} steps')

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=CFG['epochs'],
    per_device_train_batch_size=bs,
    gradient_accumulation_steps=accum,
    per_device_eval_batch_size=bs,
    prediction_loss_only=True,
    learning_rate=CFG['lr'],
    lr_scheduler_type=CFG.get('lr_scheduler', 'cosine'),
    warmup_steps=(2 if SMOKE else CFG.get('warmup_steps', 50)),
    weight_decay=CFG.get('weight_decay', 0.01),
    max_grad_norm=1.0,
    fp16=(not USE_BF16), bf16=USE_BF16,
    gradient_checkpointing=False,
    optim='adamw_torch',
    max_steps=(6 if SMOKE else -1),
    logging_steps=(1 if SMOKE else 10),
    logging_first_step=True,
    eval_strategy='steps', eval_steps=every,
    save_strategy='steps', save_steps=every,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to='none',
    disable_tqdm=True,
    remove_unused_columns=False,
    dataloader_num_workers=CFG.get('workers', 2),
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[TimeLimit(CFG.get('time_limit_h', 8.5)), Heartbeat(), NaNGuard(),
               EarlyStoppingCallback(early_stopping_patience=4)],
)

resume_ckpt = None
if CFG.get('resume', False):
    resume_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
    if resume_ckpt:
        log(f'Resume tu: {resume_ckpt}')

trainer.train(resume_from_checkpoint=resume_ckpt)

if collator.n_empty:
    log(f'CANH BAO: {collator.n_empty} mau khong tim thay vung nhan (loss mask rong).')

adapter_path = OUTPUT_DIR / 'lora_adapter'
if is_main:
    trainer.model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
log(f'Adapter da luu tai: {adapter_path}')
log(f'Tong thoi gian: {(time.time()-T0)/60:.1f} phut')
