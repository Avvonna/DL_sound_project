import torch
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(
        self,
        min_gain_in_db: float = -18.0,
        max_gain_in_db: float = 6.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str | None = None,
        sample_rate: int | None = None,
        target_rate: int | None = None,
    ):
        super().__init__()
        if max_gain_in_db < min_gain_in_db:
            raise ValueError("max_gain_in_db must be >= min_gain_in_db")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
            
        self.min_gain_in_db = float(min_gain_in_db)
        self.max_gain_in_db = float(max_gain_in_db)
        self.mode = mode
        self.p = float(p)
        # Если режим для вероятности не задан, берем такой же, как для гейна
        self.p_mode = p_mode if p_mode is not None else mode
        self.sample_rate = sample_rate
        self.target_rate = target_rate

    @staticmethod
    def _to_3d(x: Tensor):
        if x.ndim == 1:         # (T) -> (1, 1, T)
            return x[None, None, :], True, True
        
        if x.ndim == 2:         # (Channels, Time) -> (1, Channels, Time)
            # Мы считаем, что на вход подается один пример с C каналами
            return x.unsqueeze(0), True, False 
            
        if x.ndim == 3:         # (Batch, Channels, Time)
            return x, False, False
        
        raise ValueError("Expected 1D, 2D, or 3D tensor")

    @staticmethod
    def _shape_for(mode: str, B: int, C: int):
        """Определяет размерность тензора случайных чисел в зависимости от режима"""
        if mode == "per_batch":
            return (1, 1, 1)    # Один гейн на весь батч
        if mode == "per_example":
            return (B, 1, 1)    # Свой гейн для каждого примера в батче
        if mode == "per_channel":
            return (B, C, 1)    # Для каждого канала свой гейн
        raise ValueError("mode must be one of: per_batch, per_example, per_channel")

    def forward(self, x: Tensor) -> Tensor:
        # 1. Приводим всё к 3D, чтобы не мучиться с размерностями дальше
        x3, squeeze_batch, squeeze_chan = self._to_3d(x)
        B, C, T = x3.shape

        # Генерируем маску
        mask_shape = self._shape_for(self.p_mode, B, C)
        apply_mask = (torch.rand(mask_shape, device=x3.device) < self.p).to(x3.dtype)

        # Считаем гейн
        # Переводим дБ по формуле: 10^(db/20)
        gain_shape = self._shape_for(self.mode, B, C)
        gain_db = torch.empty(gain_shape, device=x3.device, dtype=x3.dtype).uniform_(
            self.min_gain_in_db, self.max_gain_in_db
        )
        gain = torch.pow(torch.tensor(10.0, device=x3.device, dtype=x3.dtype), gain_db / 20.0)

        # Смешиваем
        factor = apply_mask * gain + (1.0 - apply_mask) * 1.0
        y = x3 * factor

        # Возвращаем размерности как было в начале
        if squeeze_chan:
            y = y.squeeze(1)
        if squeeze_batch:
            y = y.squeeze(0)
        return y
