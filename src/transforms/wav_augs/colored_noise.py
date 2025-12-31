import torch
from omegaconf import ListConfig
from torch import Tensor, nn


class ColoredNoise(nn.Module):
    """
    Добавляет "цветной" шум (Colored Noise).
    Цвет зависит от спада спектральной плотности (PSD) ~ 1/f^alpha.

    alpha:
        0.0 = Белый шум (White) - равная энергия на всех частотах.
        1.0 = Розовый шум (Pink) - спад 3 дБ/октаву, естественный для слуха.
        2.0 = Коричневый шум (Brown/Red) - спад 6 дБ/октаву, глухой гул.
    """

    def __init__(
        self,
        p: float = 0.5,
        snr_db=15.0,
        sample_rate: int = 16000,
        alpha=1.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        if isinstance(snr_db, ListConfig):
            snr_db = list(snr_db)
        if isinstance(alpha, ListConfig):
            alpha = list(alpha)

        self.p = float(p)
        self.snr_db = snr_db
        self.sample_rate = int(sample_rate)
        self.alpha = alpha
        self.eps = float(eps)

    def _sample_snr_db(self, B: int, device) -> Tensor:
        v = self.snr_db
        if isinstance(v, (list, tuple)) and len(v) == 2:
            # Если передан диапазон [min, max], берем случайное значение для каждого примера
            lo, hi = float(v[0]), float(v[1])
            return (lo + (hi - lo) * torch.rand(B, device=device)).view(B, 1, 1)
        if isinstance(v, (int, float)):
            # Иначе фиксированный SNR
            return torch.full((B, 1, 1), float(v), device=device)
        raise TypeError(f"snr_db must be float/int or [min,max], got {type(v)}: {v}")

    def _sample_alpha(self, K: int, device) -> Tensor:
        v = self.alpha
        if isinstance(v, (list, tuple)) and len(v) > 0:
            # выбираем alpha для каждого примера отдельно
            choices = torch.tensor(v, device=device, dtype=torch.float32)  # (A,)
            idx = torch.randint(0, len(v), (K,), device=device)            # (K,)
            return choices[idx]                                            # (K,)
        if isinstance(v, (int, float)):
            return torch.full((K,), float(v), device=device)
        raise TypeError

    @staticmethod
    def _rms(x: Tensor, dim) -> Tensor:
        # Считаем энергию сигнала
        return torch.sqrt(torch.mean(x * x, dim=dim, keepdim=True))

    def _colored(self, white: Tensor, alpha: Tensor) -> Tensor:
        """
        Превращает белый шум в цветной через частотную фильтрацию.
        white: (Batch, Channel, Time)
        """
        # white: (K, C, T), alpha: (K,)
        K, C, T = white.shape

        # Переходим в частотный домен
        spec = torch.fft.rfft(white, dim=-1)  # (K, C, F)

        # Получаем массив частот
        # d=1.0/sr нужно, чтобы частоты были в Герцах, хотя для формулы 1/f^a масштаб не важен
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate).to(white.device)  # (F,)
        F = freqs.numel()

        # Считаем веса для фильтра
        # Амплитуда A ~ sqrt(PSD). Если PSD ~ 1/f^alpha, то A ~ 1/f^(alpha/2)
        w = torch.ones((K, F), device=white.device, dtype=white.dtype)
        logf = torch.log(freqs[1:])                     # (F-1,), freqs[0]=0 не трогаем
        a = (-alpha / 2.0).to(white.dtype).view(K, 1)   # (K, 1)
        w[:, 1:] = torch.exp(a * logf.view(1, -1))      # (K, F-1)

        # Применяем фильтр и возвращаемся во временной домен
        w = w.view(K, 1, F)                             # (K, 1, F) чтобы умножить на (K, C, F)
        spec = spec * w.to(spec.dtype)
        return torch.fft.irfft(spec, n=T, dim=-1)

    def forward(self, audio: Tensor) -> Tensor:
        # (C, T) -> (1, C, T)
        if audio.ndim == 2:
            x = audio.unsqueeze(0)
            squeeze_back = True
        else:
            x = audio
            squeeze_back = False

        B, C, T = x.shape
        device = x.device

        # Маска
        apply_mask = (torch.rand(B, device=device) < self.p)        # (B,)
        if not torch.any(apply_mask):
            return audio

        idx = torch.nonzero(apply_mask, as_tuple=False).squeeze(1)  # (K,)
        K = idx.numel()

        # Сэмплируем alpha и генерим белый шум
        alpha = self._sample_alpha(K, device=device)                # (K,)
        white = torch.randn((K, C, T), device=device, dtype=x.dtype)

        # Красим шум
        noise = self._colored(white, alpha=alpha)

        # Считаем энергии для правильного SNR
        sig_rms = self._rms(x[idx], dim=(1, 2))
        noi_rms = self._rms(noise, dim=(1, 2)) + self.eps

        snr_db = self._sample_snr_db(K, device=device)
        snr_lin = 10.0 ** (snr_db / 20.0)

        scale = (sig_rms / (noi_rms * snr_lin)).clamp_min(0.0)

        y = x.clone()
        y[idx] = x[idx] + noise * scale
        return y.squeeze(0) if squeeze_back else y
