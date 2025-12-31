import torch
from torch import Tensor, nn


class WhiteNoise(nn.Module):
    """
    DEPRECATED: использовать ColoredNoise (alpha=0.0 для белого шума).
    
    Аугментация для добавления белого шума (White Noise).
    Работает на основе целевого SNR (Signal-to-Noise Ratio).

    Args:
        p (float): Вероятность применения (0.0 - 1.0).
        snr_db (float | list | tuple): Уровень шума в децибелах.
            Если число - всегда фиксированный SNR.
            Если список [min, max] - выбираем случайно для каждого примера.
        eps (float): Маленькое число, чтобы не делить на ноль.
    """

    def __init__(self, p: float = 0.5, snr_db=15.0, eps: float = 1e-8):
        super().__init__()
        self.p = float(p)
        self.snr_db = snr_db
        self.eps = float(eps)

    def _sample_snr_db(self, B: int, device) -> Tensor:
        # Если передан диапазон [min, max], сэмплируем случайный SNR для каждого элемента батча
        if isinstance(self.snr_db, (list, tuple)) and len(self.snr_db) == 2:
            lo, hi = float(self.snr_db[0]), float(self.snr_db[1])
            # Формула: min + (max - min) * rand[0, 1]
            # view(B, 1, 1) нужен для правильного бродкастинга потом
            return (lo + (hi - lo) * torch.rand(B, device=device)).view(B, 1, 1)

        # Иначе возвращаем фиксированное значение для всего батча
        return torch.full((B, 1, 1), float(self.snr_db), device=device)

    @staticmethod
    def _rms(x: Tensor, dim) -> Tensor:
        # Считаем энергию сигнала, сохраняя размерности
        return torch.sqrt(torch.mean(x * x, dim=dim, keepdim=True))

    def forward(self, audio: Tensor) -> Tensor:
        # (Batch, Channels, Time) или (Channels, Time)
        # Если пришел один пример (C, T), добавляем размерность батча фиктивно

        if audio.ndim == 2:
            x = audio.unsqueeze(0)
            squeeze_back = True
        else:
            x = audio
            squeeze_back = False

        B, C, T = x.shape
        device = x.device

        # Генерируем маску вероятности
        apply_mask = (torch.rand((B, 1, 1), device=device) < self.p)
        if not torch.any(apply_mask):
            return audio

        # Генерируем белый шум
        noise = torch.randn_like(x)

        # Считаем энергию сигнала и шума, усредняем по каналам и времени
        sig_rms = self._rms(x, dim=(1, 2))
        noi_rms = self._rms(noise, dim=(1, 2)) + self.eps

        # Получаем целевой SNR для каждого примера
        snr_db = self._sample_snr_db(B, device=device)
        # Переводим дБ в разы (амплитуду): 10^(db/20)
        snr_lin = 10.0 ** (snr_db / 20.0)

        # Считаем коэффициент масштабирования шума
        # Формула: Scale = RMS_сигнала / (RMS_шума * Желаемое_соотношение)
        scale = (sig_rms / (noi_rms * snr_lin)).clamp_min(0.0)

        # Применяем шум
        y = x + apply_mask.float() * noise * scale

        # Если в начале добавляли измерение, убираем его обратно
        return y.squeeze(0) if squeeze_back else y
