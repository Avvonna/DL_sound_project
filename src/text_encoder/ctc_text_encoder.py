import heapq
import math
import re
from string import ascii_lowercase
from typing import List, Literal, Sequence

import torch


class CTCTextEncoder:
    """
    CTC text encoder/decoder.
    """

    EMPTY_TOK = ""
    blank_id = 0

    def __init__(self, alphabet: List[str] | None = None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {ch: i for i, ch in self.ind2char.items()}

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, item: int) -> str:
        return self.ind2char[item]

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to token indices.

        Args:
            text: Input text string.

        Returns:
            torch.Tensor: Token indices of shape [1, T] where T is text length.
        """
        text = self.normalize_text(text)
        try:
            ids = [self.char2ind[ch] for ch in text]
        except KeyError:
            unknown = sorted({ch for ch in text if ch not in self.char2ind})
            raise ValueError(f"Unknown chars: '{' '.join(unknown)}'")
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, inds: Sequence[int] | torch.Tensor) -> str:
        """Raw decoding without CTC collapsing"""
        if torch.is_tensor(inds):
            inds = inds.detach().cpu().tolist()
        return "".join([self.ind2char[int(i)] for i in inds]).strip()

    def ctc_decode(self, inds: Sequence[int] | torch.Tensor) -> str:
        if torch.is_tensor(inds):
            inds = inds.detach().cpu().tolist()

        out = []
        prev = -1
        for x in inds:
            if x != prev and x != self.blank_id:
                out.append(self.ind2char[x])
            prev = x
        return "".join(out).strip()

    @staticmethod
    def _logaddexp(a: float, b: float) -> float:
        """log(exp(a) + exp(b))"""
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a

        diff = a - b
        if diff > 20:
            return a  # exp(-20) ~ 2e-9, можно пренебречь
        if diff < -20:
            return b

        return max(a, b) + math.log1p(math.exp(-abs(diff)))

    def ctc_beam_search(
        self,
        x: torch.Tensor,
        beam_size: int = 100,
        topk_per_timestep: int | None = 20,
        beam_threshold: float = 15.0,
        input_type: Literal["log_probs", "probs"] = "log_probs",
    ) -> str:
        if input_type == "probs":
            x = torch.log(x.clamp_min(1e-10))

        log_probs = x.detach().cpu()

        return self._ctc_beam_search_logp(
            log_probs,
            beam_size=beam_size,
            topk_per_timestep=topk_per_timestep,
            beam_threshold=beam_threshold,
        )

    def _ctc_beam_search_logp(
        self,
        logp: torch.Tensor,
        beam_size: int,
        topk_per_timestep: int | None,
        beam_threshold: float,
    ) -> str:
        T, V = logp.shape

        # beam: Dict[prefix_tuple] -> (p_blank, p_non_blank)
        # Стартовое состояние
        beam = {(): (0.0, -math.inf)}

        fast_logadd = self._logaddexp
        blank_id = self.blank_id

        # Предварительная выборка topk
        use_topk = topk_per_timestep is not None and topk_per_timestep < V

        for t in range(T):
            # Текущий вектор лог-вероятностей
            row = logp[t]
            lp_blank = float(row[blank_id])

            # 1. Формируем список кандидатов для этого шага
            if use_topk:
                # +1 на случай если blank попадет в топ
                topv, topi = torch.topk(row, k=topk_per_timestep + 1)  # type: ignore

                # Фильтруем blank из кандидатов на расширение
                candidates = []
                for val, idx in zip(topv.tolist(), topi.tolist()):
                    if idx != blank_id:
                        candidates.append((idx, val))
                        if len(candidates) == topk_per_timestep:
                            break
            else:
                # Если topk нет, берем все, кроме blank
                candidates = [(i, float(row[i])) for i in range(V) if i != blank_id]

            # 2. Подготовка следующего шага
            next_beam = {}

            # Находим лучший скор в текущем луче, чтобы отсекать мусор
            # Используем max(pb, pnb) как нижнюю границу скора для скорости
            best_curr_score = -math.inf
            if beam:
                best_curr_score = max(max(pb, pnb) for pb, pnb in beam.values())

            # min_cutoff - порог, ниже которого пути не рассматриваем вообще
            min_cutoff = best_curr_score - beam_threshold

            for prefix, (p_b, p_nb) in beam.items():
                # Если путь уже слишком плохой, пропускаем его расширение
                if p_b < min_cutoff and p_nb < min_cutoff:
                    continue

                # Считаем полную вероятность пути
                p_total = fast_logadd(p_b, p_nb)

                # A. Шаг Blank
                # Путь не меняется, вероятность = p_total + prob(blank)
                n_p_b, n_p_nb = next_beam.get(prefix, (-math.inf, -math.inf))
                n_p_b = fast_logadd(n_p_b, p_total + lp_blank)
                next_beam[prefix] = (n_p_b, n_p_nb)

                # B. Шаг расширения (Non-Blank)
                last_char = prefix[-1] if prefix else None

                for s_idx, s_prob in candidates:
                    # s_prob уже float

                    # Случай 1: Повтор символа (aa -> a)
                    if s_idx == last_char:
                        # 1.1 Если мы пришли из blank (a -> _ -> a), то это новый символ: "aa"
                        # Вероятность: p_b + s_prob
                        new_prefix = prefix + (s_idx,)
                        n_pb_new, n_pnb_new = next_beam.get(
                            new_prefix, (-math.inf, -math.inf)
                        )
                        n_pnb_new = fast_logadd(n_pnb_new, p_b + s_prob)
                        next_beam[new_prefix] = (n_pb_new, n_pnb_new)

                        # 1.2 Если мы пришли из non-blank (a -> a), то это схлопывание: "a"
                        # Вероятность: p_nb + s_prob
                        # Префикс не меняется (остается prefix)
                        n_pb_curr, n_pnb_curr = next_beam[
                            prefix
                        ]  # уже создан выше в шаге A
                        n_pnb_curr = fast_logadd(n_pnb_curr, p_nb + s_prob)
                        next_beam[prefix] = (n_pb_curr, n_pnb_curr)

                    # Случай 2: Новый символ (a -> b)
                    else:
                        new_prefix = prefix + (s_idx,)
                        n_pb_new, n_pnb_new = next_beam.get(
                            new_prefix, (-math.inf, -math.inf)
                        )
                        n_pnb_new = fast_logadd(n_pnb_new, p_total + s_prob)
                        next_beam[new_prefix] = (n_pb_new, n_pnb_new)

            if len(next_beam) > beam_size:
                # Чтобы не считать logaddexp для всех, используем max(pb, pnb) как приближение
                # og(e^a + e^b) ≈ max(a, b), ошибка мала.
                beam = dict(
                    heapq.nlargest(
                        beam_size,
                        next_beam.items(),
                        key=lambda x: max(x[1][0], x[1][1]),
                    )
                )
            else:
                beam = next_beam

        best_prefix = max(
            beam.items(),
            key=lambda kv: fast_logadd(kv[1][0], kv[1][1]),
        )[0]

        return self.decode(best_prefix)
