from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, num_rnn_layers=3):
        super().__init__()

        # Сверточный блок:
        # - уменьшает время в 2 раза на каждом слое с stride=(2, 1)
        # - уменьшает частоты в 2 раза на каждом слое с stride=(2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )

        # Conv1: H_out = floor((n_feats + 2*20 - 41)/2 + 1) = floor(n_feats/2)
        # Conv2: H_out = floor((H_out + 2*10 - 21)/2 + 1) = floor(H_out/2)

        rnn_input_size = n_feats
        rnn_input_size = (rnn_input_size + 2 * 20 - 41) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size *= 32  # channels

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(rnn_hidden * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # spectrogram: (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)

        # Conv
        x = self.conv(x)

        # (B, C, F_new, T_new) -> (B, T_new, C * F_new)
        B, C, F_new, T_new = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_new, -1)

        # Recalculate lengths
        new_lengths = self.transform_input_lengths(spectrogram_length)

        # Pack padded sequence
        x = pack_padded_sequence(
            x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        output = self.fc(x)
        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": new_lengths}

    def transform_input_lengths(self, input_lengths):
        # L_out = (L_in + 2*padding - kernel) // stride + 1

        # Conv 1
        # kernel=(41, 11), stride=(2, 2), padding=(20, 5)
        new_lengths = (input_lengths + 2 * 5 - 11) // 2 + 1

        # Conv 2
        # kernel=(21, 11), stride=(2, 1), padding=(10, 5)
        new_lengths = (new_lengths + 2 * 5 - 11) // 1 + 1

        return new_lengths

class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, n_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            hidden_size * 2,    # вход = 2*hidden_size
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.ln = nn.LayerNorm(hidden_size * 2) # Bidirectional удваивает выход
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.ln(out)
        out = self.dropout(out)

        return x + out

class UpdDeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, num_rnn_layers=5):
        super().__init__()

        # CNN: Stride 2x2 в обоих слоях -> Time / 4
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )

        # Расчет выхода после CNN
        f1 = (n_feats + 2*20 - 41)//2 + 1
        f2 = (f1 + 2*10 - 21)//2 + 1
        rnn_input_dim = 32 * f2

        # Проекция CNN выхода в размерность RNN
        self.proj = nn.Linear(rnn_input_dim, rnn_hidden * 2) # *2 так как BiGRU

        # RNN Layers с LayerNorm и Residuals
        layers = []
        for _ in range(num_rnn_layers):
            layers.append(ResidualGRU(rnn_hidden, dropout=0.1))
        self.rnn_layers = nn.ModuleList(layers)

        # Head
        self.fc = nn.Linear(rnn_hidden * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # spectrogram: (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)

        # Conv
        x = self.conv(x)

        # Размеры
        B, C, F, T = x.shape
        # (B, C, F, T) -> (B, T, C*F)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, -1)

        # Пересчет длин
        new_lengths = self.transform_input_lengths(spectrogram_length)

        # Проекция, чтобы размерности совпадали для residual connection
        x = self.proj(x)

        # RNN Loop
        for layer in self.rnn_layers:
            x = layer(x, new_lengths)

        output = self.fc(x)
        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": new_lengths}

    def transform_input_lengths(self, input_lengths):
        # Conv 1: stride (2, 2) -> Time / 2
        lengths = (input_lengths + 2 * 5 - 11) // 2 + 1
        # Conv 2: stride (2, 2) -> Time
        lengths = (lengths + 2 * 5 - 11) // 2 + 1
        return lengths
