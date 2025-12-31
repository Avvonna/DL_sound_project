import io

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(Image.open(buf))

    plt.close()

    return image

def _spec_to_2d(spec):
    """
    Приводит спектрограмму к 2D виду [F, T].
    Поддерживает [F, T], [1, F, T], [F, T, 1].
    """
    if torch.is_tensor(spec):
        spec = spec.detach().cpu()

    if spec.dim() == 3:
        if spec.shape[0] == 1:          # [1, F, T]
            spec = spec[0]
        elif spec.shape[-1] == 1:       # [F, T, 1]
            spec = spec[..., 0]
        else:
            raise ValueError(f"Ожидалась спектрограмма [F,T] или [1,F,T], получили {tuple(spec.shape)}")

    if spec.dim() != 2:
        raise ValueError(f"Ожидалась 2D спектрограмма [F,T], получили {tuple(spec.shape)}")

    return spec

def plot_spectrogram(spectrogram, name=None):
    """
    Plot spectrogram

    Args:
        spectrogram (Tensor): spectrogram tensor.
        name (None | str): optional name.
    Returns:
        image (Image): image of the spectrogram
    """
    spectrogram = _spec_to_2d(spectrogram)

    fig = plt.figure(figsize=(20, 5))
    plt.pcolormesh(spectrogram)
    if name:
        plt.title(name)
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    # convert buffer to Tensor
    image = ToTensor()(Image.open(buf))

    plt.close(fig)
    return image

def plot_spectrogram_grid(spectrograms, titles=None, figsize=(20, 16)):
    """
    Рисует несколько спектрограмм в одной фигуре (в столбик).

    Args:
        spectrograms (list[Tensor]): список спектрограмм
        titles (list[str] | None): заголовки для каждой спектрограммы.
        figsize (tuple): размер фигуры.

    Returns:
        image (Tensor): объединенное изображение (C x H x W).
    """
    n = len(spectrograms)
    if titles is None:
        titles = [None] * n

    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, spec, title in zip(axes, spectrograms, titles):
        spec = _spec_to_2d(spec)
        ax.pcolormesh(spec)
        if title:
            ax.set_title(title)
        ax.axis("off")

    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(Image.open(buf))
    plt.close(fig)
    return image
