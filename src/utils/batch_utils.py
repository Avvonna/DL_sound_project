from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

import torch


def move_tensors_to_device(batch: Dict[str, Any], device: str, tensor_names: Iterable[str]) -> Dict[str, Any]:
    """
    Переносит на device только те поля батча,  
    которые перечислены в tensor_names и являются torch.Tensor.

    Args:
        batch: Батч от DataLoader (dict).
        device: Устройство.
        tensor_names: Имена полей батча, которые нужно переносить.

    Returns:
        Тот же dict batch (мутируется на месте), с перенесенными тензорами.
    """

    for name in tensor_names:
        if name in batch and torch.is_tensor(batch[name]):
            batch[name] = batch[name].to(device, non_blocking=True)
    return batch


def apply_transforms_to_tensors(
    batch: Dict[str, Any],
    transforms: Dict[str, Callable] | None,
) -> Dict[str, Any]:
    """
    Применяет batch-трансформы к тензорным полям батча.

    Ожидается, что transforms - это отображение вида:
        {"spectrogram": nn.Module, "text_encoded": nn.Module, ...}

    Args:
        batch: Батч от DataLoader (dict).
        transforms: Отображение "имя_тензора -> nn.Module" или None.

    Returns:
        Тот же dict batch с преобразованными тензорами.
    """
    # do batch transforms on device
    if not transforms:
        return batch

    for tensor_name, transform in transforms.items():
        if tensor_name in batch and torch.is_tensor(batch[tensor_name]):
            batch[tensor_name] = transform(batch[tensor_name])
    return batch
