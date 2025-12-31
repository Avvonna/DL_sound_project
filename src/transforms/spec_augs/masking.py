import torch
import torchaudio
from torch import nn


class _BaseMasking(nn.Module):
    """
    Base class for time/frequency masking that correctly supports:
      - (F, T)
      - (C, F, T)
      - (B, F, T)
      - (B, C, F, T)

    Important:
      - Applies masking per-sample when batch dimension exists (B).
      - Probability p is applied independently per sample.
      - Preserves input shape.
    """

    aug: nn.Module

    def __init__(self, p: float):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.aug = nn.Identity()  # default (changed in subclasses)

    def _apply_one(self, x_3d: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to one sample in (C, F, T) format.
        """
        return self.aug(x_3d)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if spectrogram.dim() not in (2, 3, 4):
            raise ValueError(
                "spectrogram must have shape (F,T), (C,F,T), (B,F,T), or (B,C,F,T); "
                f"got {tuple(spectrogram.shape)}"
            )

        # (F, T) -> (1, F, T)
        if spectrogram.dim() == 2:
            if torch.rand(1, device=spectrogram.device).item() > self.p:
                return spectrogram
            x = spectrogram.unsqueeze(0)
            x = self._apply_one(x)
            return x.squeeze(0)

        # dim == 3: считаем, что это (B, F, T)
        if spectrogram.dim() == 3:
            B, F, T = spectrogram.shape
            out = spectrogram
            modified = False

            for i in range(B):
                if torch.rand(1, device=spectrogram.device).item() > self.p:
                    continue
                if not modified:
                    out = spectrogram.clone()
                    modified = True
                out[i] = self._apply_one(out[i].unsqueeze(0)).squeeze(0)  # (1,F,T)->(F,T)

            return out

        # dim == 4: (B,C,F,T) -> per sample
        B, C, F, T = spectrogram.shape
        out = spectrogram
        modified = False
        for i in range(B):
            if torch.rand(1, device=spectrogram.device).item() > self.p:
                continue
            if not modified:
                out = spectrogram.clone()
                modified = True
            out[i] = self._apply_one(out[i])
        return out


class FrequencyMasking(_BaseMasking):
    def __init__(self, p: float = 0.5, freq_mask_param: int = 15):
        super().__init__(p=p)
        self.aug = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )


class TimeMasking(_BaseMasking):
    def __init__(self, p: float = 0.5, time_mask_param: int = 35):
        super().__init__(p=p)
        self.aug = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
