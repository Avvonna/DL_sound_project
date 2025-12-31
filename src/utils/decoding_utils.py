from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DecodingConfig:
    """
    Конфигурация декодирования CTC.

    Поля:
        decode_type: Режим декодирования ("greedy" или "beam").
        beam_size: Ширина луча для beam search.
        topk_per_timestep: Ограничение числа кандидатов на каждом шаге времени.
        beam_threshold: Порог отсечения плохих гипотез относительно лучшей.
        save_both_decodes: Сохранять ли одновременно greedy и beam.
    """
    decode_type: str = "greedy"          # "greedy" | "beam"
    beam_size: int = 10
    topk_per_timestep: Optional[int] = None
    beam_threshold: float = 70.0
    save_both_decodes: bool = False


def parse_decoding_cfg(cfg) -> DecodingConfig:
    """
    Парсит секцию `decoding` из Hydra/OmegaConf-конфига (или обычного dict).

    Замечания:
        - Функция специально приводит типы (int/float/bool), чтобы дальше не тащить
          по коду OmegaConf-обертки и избежать неожиданных типов.
        - Если cfg=None или секции `decoding` нет, берутся значения по умолчанию.
    """
    d = cfg.get("decoding", {}) if cfg is not None else {}
    tpt = d.get("topk_per_timestep", None)
    return DecodingConfig(
        decode_type=str(d.get("decode_type", "greedy")),
        beam_size=int(d.get("beam_size", 10)),
        topk_per_timestep=int(tpt) if tpt is not None else None,
        beam_threshold=float(d.get("beam_threshold", 70.0)),
        save_both_decodes=bool(d.get("save_both_decodes", False)),
    )


def decode_greedy(
    text_encoder,
    log_probs: torch.Tensor,    # [B, T, V] log-softmax по словарю
    lengths: torch.Tensor      # [B] реальные длины по времени (до паддинга)
) -> list[str]:
    """
    Greedy-декодирование для CTC: argmax по словарю на каждом таймстепе + CTC-collapse.

    Алгоритм:
        1) pred_ids = argmax(log_probs, dim=-1) -> [B, T]
        2) для каждого примера i берем первые lengths[i] таймстепов
        3) text_encoder.ctc_decode схлопывает повторы и удаляет blank

    Возвращает:
        Список строк длиной B.
    """
    pred_ids = log_probs.argmax(dim=-1).detach().cpu()
    lengths_cpu = lengths.detach().cpu().tolist()

    out: list[str] = []
    for seq, L in zip(pred_ids, lengths_cpu):
        out.append(text_encoder.ctc_decode(seq[: int(L)].tolist()))
    return out


def decode_beam(
    text_encoder,
    log_probs: torch.Tensor,            # [B, T, V] log-softmax
    lengths: torch.Tensor,              # [B]
    beam_size: int,
    topk_per_timestep: int | None,
    beam_threshold: float,
) -> list[str]:
    """
    Beam search декодирование для CTC (в log-space).

    Замечания:
        - Для корректности и стабильности используем log_probs (не probs).
        - Для ускорения можно ограничить topk_per_timestep: на каждом таймстепе
          рассматриваются только top-k токенов (кроме blank).

    Возвращает:
        Список строк длиной B.
    """
    lp = log_probs.detach().cpu()
    lengths_cpu = lengths.detach().cpu().tolist()

    out: list[str] = []
    for i, L in enumerate(lengths_cpu):
        out.append(
            text_encoder.ctc_beam_search(
                lp[i, : int(L)],        # [T, V] без паддинга
                beam_size=beam_size,
                topk_per_timestep=topk_per_timestep,
                beam_threshold=beam_threshold,
                input_type="log_probs",
            )
        )
    return out
