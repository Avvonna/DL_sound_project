import torch
import torchaudio
from torch import Tensor, nn


class SpeedPerturb(nn.Module):
    """
    Класс для изменения скорости аудио (Speed Perturbation).
    Работает через ресемплинг: меняем частоту дискретизации, но притворяемся,
    что она осталась прежней. Из-за этого меняется и высота тона (Pitch), и длительность.

    Чтобы собрать батч обратно, приходится обрезать или паддить (дополнять нулями)
    сигнал до исходной длины.
    """

    def __init__(
        self,
        p: float = 0.5,
        sample_rate: int = 16000,
        speeds=(0.9, 1.0, 1.1),
    ):
        super().__init__()
        self.p = float(p)
        self.sample_rate = int(sample_rate)
        self.speeds = tuple(float(s) for s in speeds)

        # Создаём ресемплеры
        self._resamplers = {}
        for s in self.speeds:
            if s == 1.0:
                continue
            new_sr = int(round(self.sample_rate / s))
            self._resamplers[s] = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate, new_freq=new_sr
            )

    def _fix_length(self, y: Tensor, T: int) -> Tensor:
        # Функция, чтобы подогнать длину аудио под исходную T.
        # y: (Channels, Time_new)
        T2 = y.shape[-1]

        # Если стало длиннее, обрезаем хвост
        if T2 > T:
            return y[..., :T]

        # Если стало короче, добиваем нулями справа
        if T2 < T:
            pad = T - T2
            return torch.nn.functional.pad(y, (0, pad))

        return y

    def forward(self, audio: Tensor) -> Tensor:
        if torch.rand(()) >= self.p:
            return audio

        # (Channels, Time) -> (1, C, T)
        if audio.ndim == 2:
            x = audio.unsqueeze(0)
            squeeze_back = True
        else:
            x = audio
            squeeze_back = False

        B, C, T = x.shape

        out = []
        # Тут приходится делать цикл for по батчу...
        for b in range(B):
            s = self.speeds[int(torch.randint(0, len(self.speeds), (1,)).item())]

            # Если скорость 1.0, то просто копируем
            if s == 1.0:
                y = x[b]
            else:
                y = self._resamplers[s](x[b])

            # Подгоняем размер
            y = self._fix_length(y, T)
            out.append(y)

        # Собираем в один батч
        y = torch.stack(out, dim=0)

        # Если на входе не было батча, убираем лишнюю размерность
        return y.squeeze(0) if squeeze_back else y
