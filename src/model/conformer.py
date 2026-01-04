import numpy as np
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

def conv2d_out_len(L: int, k: int = 3, s: int = 2, p: int = 1) -> int:
    """
    Выходная длина/ширина для Conv2d по одной оси:
    L_out = floor((L + 2p - k)/s) + 1
    """
    return (L + 2 * p - k) // s + 1

class ConvSubsampling(nn.Module):
    """
    2x Conv2d со stride=2: уменьшает время в ~4 раза (и частоты тоже),
    затем "склеивает" (каналы * частоты) в один признак.

    Вход:  (B, F, T)
    Выход: (B, T', C*F')
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T) -> (B, 1, F, T)
        x = x.unsqueeze(1)

        # (B, C, F', T')
        x = self.conv(x)
        b, c, f, t = x.size()

        # (B, C, F', T') -> (B, T', C, F') -> (B, T', C*F')
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        return x

class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe: (max_len, d_model), будет автоматически на нужном device/dtype
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[: x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class ConformerBlock(nn.Module):
    """
    Упрощенный Conformer-like блок:
    SelfAttention -> ConvModule -> FeedForward (везде pre-norm + residual)

    Вход/выход: (B, T, D)
    """
    def __init__(self, d_model: int, n_head: int, conv_kernel_size: int, dropout: float = 0.1):
        super().__init__()

        # MHSA
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Conv module (depthwise)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),  # pointwise
            nn.GLU(dim=1),                                   # (B, 2D, T) -> (B, D, T)
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=d_model,                              # depthwise
            ),
            nn.BatchNorm1d(d_model),
            Swish(),
            nn.Conv1d(d_model, d_model, kernel_size=1),      # pointwise
            nn.Dropout(dropout),
        )

        # FFN
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)

        # MHSA + residual
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = residual + self.dropout1(x)

        # Conv + residual
        residual = x
        x = self.layer_norm2(x).transpose(1, 2)  # (B, D, T)
        x = self.conv_module(x).transpose(1, 2)  # (B, T, D)
        x = residual + x

        # FFN + residual
        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = residual + x

        return x

class Conformer(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_head: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Subsampling: (B, F, T) -> (B, T', d_model*F')
        self.subsampling = ConvSubsampling(out_channels=d_model)

        freq = n_feats
        for _ in range(2):
            freq = conv2d_out_len(freq, k=3, s=2, p=1)

        # Проекция (d_model*F') -> d_model
        self.linear_proj = nn.Linear(d_model * freq, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Blocks
        self.layers = nn.ModuleList(
            [ConformerBlock(d_model, n_head, conv_kernel_size, dropout) for _ in range(n_layers)]
        )

        # Head
        self.fc = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length, **batch):
        """
        spectrogram: (B, F, T)

        Возвращаем:
          log_probs: (B, T', C)  (batch_first)
        """
        x = self.subsampling(spectrogram)  # (B, T', d_model*F')
        x = self.linear_proj(x)            # (B, T', d_model)
        x = self.positional_encoding(x)    # (B, T', d_model)

        for layer in self.layers:
            x = layer(x)

        logits = self.fc(x)               # (B, T', n_tokens)
        log_probs = logits.log_softmax(dim=-1)

        new_lengths = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": new_lengths}

    def transform_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Корректировка длин по времени после двух Conv2d (kernel=3, stride=2, padding=1).
        Нужна для CTC (input_lengths должны соответствовать T').

        input_lengths: (B,) длины по T
        return:        (B,) длины по T'
        """
        out = input_lengths
        for _ in range(2):
            out = (out + 2 * 1 - 3) // 2 + 1  # тот же conv2d_out_len, но для тензора
        return out
