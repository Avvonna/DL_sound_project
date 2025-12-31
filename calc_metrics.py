from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from src.metrics.utils import calc_cer, calc_wer, get_distance
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def collect_pairs(gt_dir: Path, pred_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    # используем имя файла без расширения
    gt_files = {p.stem: p for p in gt_dir.glob("*.txt") if p.is_file()}
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt") if p.is_file()}

    common = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    return {k: (gt_files[k], pred_files[k]) for k in common}


def main():
    parser = argparse.ArgumentParser()

    # Директория с настоящими транскрипциями (*.txt)
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Path to ground-truth transcriptions (*.txt)",
    )

    # Директория с предсказаниями (*.txt)
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Path to predicted transcriptions (*.txt)",
    )

    # (Опционально) Директория, куда складывать метрики
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to save per-utt metrics as json",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    # Собираем пары файлов gt/pred
    pairs = collect_pairs(gt_dir, pred_dir)
    if len(pairs) == 0:
        raise RuntimeError(
            "No matching *.txt files by stem between gt_dir and pred_dir"
        )

    # Нормализация текста
    normalize = CTCTextEncoder.normalize_text

    total_cer = 0.0
    total_wer = 0.0

    cer_edits = 0
    cer_den = 0
    wer_edits = 0
    wer_den = 0

    details = {}

    for utt_id, (gt_path, pr_path) in pairs.items():
        gt = normalize(read_text(gt_path))
        pr = normalize(read_text(pr_path))

        cer = calc_cer(gt, pr)
        wer = calc_wer(gt, pr)

        # micro (global) accumulators
        if gt:
            cer_edits += int(get_distance(gt, pr))
            cer_den += len(gt)

        gt_w = gt.split()
        pr_w = pr.split()
        if gt_w:
            wer_edits += int(get_distance(gt_w, pr_w))
            wer_den += len(gt_w)

        total_cer += cer
        total_wer += wer

        details[utt_id] = {
            "cer": cer,
            "wer": wer,
            "gt": gt,
            "pred": pr,
        }

    avg_cer = total_cer / len(pairs)
    avg_wer = total_wer / len(pairs)

    micro_cer = (cer_edits / cer_den) if cer_den > 0 else float("nan")
    micro_wer = (wer_edits / wer_den) if wer_den > 0 else float("nan")

    print(f"Matched files: {len(pairs)}")
    print(f"CER macro:  {avg_cer:.6f} ({avg_cer*100:.2f}%)")
    print(f"WER macro:  {avg_wer:.6f} ({avg_wer*100:.2f}%)")
    print(f"CER micro:  {micro_cer:.6f} ({micro_cer*100:.2f}%)")
    print(f"WER micro:  {micro_wer:.6f} ({micro_wer*100:.2f}%)")

    # Сохранение метрик
    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved details to: {out_path}")


if __name__ == "__main__":
    main()
