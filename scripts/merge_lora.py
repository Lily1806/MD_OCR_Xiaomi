#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_lora.py
------------------------------------------------------------------------------------------
Ket hop hai hoac nhieu LoRA adapter bang trung binh co trong so (weighted average).
Ket qua la mot adapter moi co the dung thay the bat ky adapter thanh phan nao.

Vi du su dung:
    # Tron 70% adapter_v9 va 30% adapter_v15 -> merged_7030
    python3 merge_lora.py \
        --adapters adapter_v9 adapter_v15 \
        --weights  0.7 0.3 \
        --output   merged_lora_7030

    # Tron 3 adapter voi trong so bat ki (tu dong chuan hoa)
    python3 merge_lora.py \
        --adapters a1 a2 a3 \
        --weights  0.5 0.3 0.2 \
        --output   merged_3way
------------------------------------------------------------------------------------------
"""

import argparse
import json
import shutil
from pathlib import Path

try:
    import safetensors.torch as st
    import torch
except ImportError as e:
    raise ImportError(
        "Thieu thu vien. Hay cai dat: pip install safetensors torch"
    ) from e


def load_adapter(adapter_dir: Path) -> dict:
    """Nap adapter_model.safetensors va tra ve dict tensor."""
    sf_path = adapter_dir / "adapter_model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(f"Khong tim thay: {sf_path}")
    return st.load_file(str(sf_path))


def merge_adapters(
    adapter_dirs: list,
    weights: list,
    output_dir: Path,
) -> None:
    """
    Tinh trung binh co trong so cac tensor tu nhieu adapter va luu ra output_dir.

    Tat ca adapter phai co cung tap khoa (layer names).
    Weights se duoc chuan hoa tu dong de tong = 1.0.
    """
    assert len(adapter_dirs) == len(weights), "So luong adapter va weights phai bang nhau."
    assert len(adapter_dirs) >= 2, "Can it nhat 2 adapter de merge."

    # Chuan hoa trong so
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]
    print(f"Trong so sau chuan hoa: {[round(w, 4) for w in norm_weights]}")

    # Nap tat ca adapter
    print("Dang nap adapter...")
    tensors_list = []
    for d in adapter_dirs:
        t = load_adapter(d)
        tensors_list.append(t)
        print(f"  Da nap: {d}  ({len(t)} layer)")

    # Kiem tra khoa khop nhau
    keys_0 = set(tensors_list[0].keys())
    for i, t in enumerate(tensors_list[1:], 1):
        diff = keys_0.symmetric_difference(set(t.keys()))
        if diff:
            print(f"  Canh bao: Adapter {i} co {len(diff)} khoa khac biet: {list(diff)[:5]}")

    # Tinh trung binh co trong so
    print("Dang tinh trung binh...")
    merged = {}
    for key in keys_0:
        acc = None
        for t, w in zip(tensors_list, norm_weights):
            if key not in t:
                continue
            tensor = t[key].float()
            if acc is None:
                acc = tensor * w
            else:
                acc = acc + tensor * w
        if acc is not None:
            merged[key] = acc.to(tensors_list[0][key].dtype)

    # Luu ket qua
    output_dir.mkdir(parents=True, exist_ok=True)
    out_sf = output_dir / "adapter_model.safetensors"
    st.save_file(merged, str(out_sf))
    print(f"Da luu merged adapter: {out_sf}  ({out_sf.stat().st_size / 1024**2:.1f} MB)")

    # Copy adapter_config.json tu adapter dau tien
    cfg_src = adapter_dirs[0] / "adapter_config.json"
    cfg_dst = output_dir / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy(str(cfg_src), str(cfg_dst))
        print(f"Da copy adapter_config.json tu {adapter_dirs[0]}")
    else:
        print("Canh bao: Khong tim thay adapter_config.json. Hay copy thu cong.")

    print(f"\nHoan tat. Adapter merged tai: {output_dir}")
    print(f"Su dung adapter nay bang cach thay the duong dan adapter_dir trong notebook huan luyen.")


def main():
    parser = argparse.ArgumentParser(description="Merge nhieu LoRA adapter bang weighted average.")
    parser.add_argument(
        "--adapters", nargs="+", required=True,
        help="Duong dan toi cac thu muc adapter (theo thu tu)",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, required=True,
        help="Trong so tuong ung voi tung adapter (tu dong chuan hoa)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Duong dan thu muc dau ra cho adapter merged",
    )
    args = parser.parse_args()

    adapter_dirs = [Path(d) for d in args.adapters]
    output_dir   = Path(args.output)

    for d in adapter_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Thu muc adapter khong ton tai: {d}")

    merge_adapters(adapter_dirs, args.weights, output_dir)


if __name__ == "__main__":
    main()

