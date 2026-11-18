import os, sys, json, traceback
sys.path.insert(0, os.environ.get('CC_WORK', '/kaggle/working'))
import ocr_common as cc
cc.prepare_env()

import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
cc.patch_peft_torchao()

CFG     = json.load(open(os.environ['CFG_PATH']))
N_REQ   = int(os.environ.get('EVAL_N', '100'))
OUT     = os.environ.get('EVAL_OUT')
DTYPE   = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
ADAPTER = Path(CFG['output_dir']) / 'lora_adapter'


def cer(gt, pred):
    if not gt:
        return 0.0 if not pred else 1.0
    m, n = len(gt), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if gt[i-1] == pred[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n] / m


def main():
    processor = AutoProcessor.from_pretrained(CFG['model_id'], trust_remote_code=True)
    val_ds = cc.OCRDataset(Path(CFG['dataset_dir']) / 'val.jsonl', CFG['dataset_dir'],
                           prompt_override=CFG.get('short_prompt'))
    try:
        base = AutoModelForImageTextToText.from_pretrained(
            CFG['model_id'], dtype=DTYPE, device_map={'': 0}, trust_remote_code=True)
    except TypeError:
        base = AutoModelForImageTextToText.from_pretrained(
            CFG['model_id'], torch_dtype=DTYPE, device_map={'': 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    max_new = int(CFG.get('max_new', 24))

    def predict_one(rec):
        image = Image.open(val_ds.resolve(rec['image'])).convert('RGB')
        try:
            text = processor.apply_chat_template(
                cc.clean_messages(rec['messages'][:-1]), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = processor.apply_chat_template(
                cc.clean_messages(rec['messages'][:-1]), tokenize=False,
                add_generation_prompt=True)
        inp = processor(text=text, images=[image], return_tensors='pt')
        inp = {k: (v.to(model.device, dtype=DTYPE) if v.is_floating_point() else v.to(model.device))
               for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=True)
        return processor.tokenizer.decode(
            out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    N = min(N_REQ, len(val_ds.records))
    n_exact = 0; total_cer = 0.0; rows = []
    for i, rec in enumerate(val_ds.records[:N]):
        gt   = cc.assistant_text(rec)
        pred = predict_one(rec)
        ok   = (pred.strip() == gt.strip())
        c    = cer(gt, pred)
        n_exact += ok; total_cer += c
        rows.append({'image': rec['image'], 'gt': gt, 'pred': pred, 'ok': ok, 'cer': round(c, 3)})
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{N}] exact={n_exact/(i+1)*100:.1f}%  CER={total_cer/(i+1)*100:.1f}%', flush=True)

    print('\n' + '=' * 56)
    print(f'Exact match: {n_exact/N*100:.2f}%  ({N} o val)')
    print(f'Average CER: {total_cer/N*100:.2f}%')
    wrong = [r for r in rows if not r['ok']]
    print(f'So o sai: {len(wrong)}/{N}')
    for r in wrong[:8]:
        print(f'  GT: {r["gt"][:60]!r}  PRED: {r["pred"][:60]!r}  CER={r["cer"]:.2f}')
    print('=' * 56)
    if OUT:
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'Saved -> {OUT}')

    # === ĐÁNH GIÁ TẬP 720 Ô M/D THỰC TẾ TRÊN GPU ===
    eval_720_path = Path("/tmp/dataset/eval_720/eval_720.jsonl")
    if eval_720_path.exists():
        print("\n" + "=" * 60)
        print("🚀 BẮT ĐẦU ĐÁNH GIÁ 720 Ô M/D TRÊN GPU...")
        print("=" * 60)
        processor.tokenizer.padding_side = 'right'  # TUYỆT ĐỐI không padding ở đầu
        
        with open(eval_720_path, encoding='utf-8') as f:
            recs_720 = [json.loads(line) for line in f if line.strip()]
        
        n_720 = len(recs_720)
        ok_720 = 0
        cer_720 = 0.0
        rows_720 = []
        
        for idx, r in enumerate(recs_720):
            sid = r["id"]
            gt = r["gt"]
            img_f = Path("/tmp/dataset") / r["image"]
            try:
                im = Image.open(img_f).convert("RGB")
                eval_prompt = CFG.get("short_prompt", "Transcribe the handwritten text in this cell. Output only the text.")
                text_prompt = processor.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": eval_prompt}
                    ]}],
                    tokenize=False, add_generation_prompt=True
                )
                inp = processor(text=text_prompt, images=[im], return_tensors="pt")
                inp = {k: (v.to(model.device, dtype=DTYPE) if v.is_floating_point() else v.to(model.device))
                       for k, v in inp.items()}
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=24, do_sample=False, use_cache=True)
                pred = processor.tokenizer.decode(
                    out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            except Exception as e:
                pred = f"__ERR__: {e}"
            
            is_match = (pred.strip() == gt.strip())
            c_dist = cer(gt, pred)
            ok_720 += is_match
            cer_720 += c_dist
            rows_720.append({
                "id": sid,
                "image": r["image"],
                "gt": gt,
                "pred": pred,
                "exact_match": is_match,
                "cer": round(c_dist, 4)
            })
            
            if (idx + 1) % 100 == 0 or (idx + 1) == n_720:
                print(f"  [{idx+1}/{n_720}] Exact: {ok_720/(idx+1)*100:.2f}% | CER: {cer_720/(idx+1)*100:.2f}%", flush=True)

        em_pct_720 = ok_720 / n_720 * 100
        avg_cer_720 = cer_720 / n_720 * 100
        print("\n" + "=" * 60)
        print(f"🎉 KẾT QUẢ 720 Ô M/D: {ok_720}/{n_720} ({em_pct_720:.2f}%)")
        print(f"   CER trung bình: {avg_cer_720:.2f}%")
        print("=" * 60)
        
        pred_out_file = Path(CFG["output_dir"]) / "predictions_720.json"
        with open(pred_out_file, "w", encoding="utf-8") as f:
            json.dump(rows_720, f, ensure_ascii=False, indent=2)
            
        sum_out_file = Path(CFG["output_dir"]) / "summary_720.json"
        with open(sum_out_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": n_720,
                "exact_match_count": ok_720,
                "exact_match_pct": em_pct_720,
                "avg_cer_pct": avg_cer_720,
                "failures_count": n_720 - ok_720
            }, f, indent=2)
        print(f"Đã lưu kết quả đánh giá 720 ô -> {pred_out_file}")


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
