import torch
from torch import Tensor, nn


class CTCLossWrapper(nn.Module):
    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: Tensor,            # ожидается (B, T, C)
        log_probs_length: Tensor,     # (B,)
        text_encoded: Tensor,         # (N,S) или (sum(target_lengths),)
        text_encoded_length: Tensor,  # (B,)
        **batch,
    ) -> dict[str, Tensor]:
        # CTCLoss ожидает (T, B, C)
        if log_probs.dim() != 3:
            raise ValueError(f"log_probs must be 3D, got shape={tuple(log_probs.shape)}")

        log_probs_t = log_probs.transpose(0, 1)  # (T, B, C)

        input_lengths = log_probs_length.to(dtype=torch.long).cpu()
        target_lengths = text_encoded_length.to(dtype=torch.long).cpu()

        # targets обычно удобно держать на том же device, что и log_probs
        targets = text_encoded.to(dtype=torch.long, device=log_probs.device)

        loss = self.ctc(
            log_probs=log_probs_t,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        return {"loss": loss}
