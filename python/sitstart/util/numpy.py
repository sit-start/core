import os
from tempfile import NamedTemporaryFile
from typing import cast

import imageio
import imageio_ffmpeg  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from IPython.display import display
from ipywidgets import Video
from PIL import Image

os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"


def imshow(img: np.ndarray, height: float = 4.0) -> None:
    fig = plt.gcf()
    fig.set_figheight(height)
    fig.set_figwidth(height * img.shape[1] / img.shape[0])
    ax = fig.add_subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


# see PIL.Image.resize for additional arguments
def imresize(arr: np.ndarray, size: tuple[int, int], **kwargs) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize(size, **kwargs))


# @source: https://github.com/znah/notebooks/blob/master/external_colab_snippets.ipynb
def imtile(images: np.ndarray | torch.Tensor, cols: int | None = None):
    # TODO: implement in torch so it stays on the device
    is_torch = isinstance(images, torch.Tensor)
    device = None
    if is_torch:
        device = images.device
        images = images.permute(0, 2, 3, 1).cpu().numpy()

    N, H, W, _ = images.shape
    images = np.asarray(images)
    if cols is None:
        cols = int(np.ceil(np.sqrt(N)))
    H, W = images.shape[1:3]
    pad = (cols - N) % cols
    images = np.pad(
        images, np.array([(0, pad)] + [(0, 0)] * (images.ndim - 1)), "constant"
    )
    rows = len(images) // cols
    images = images.reshape(rows, cols, *images.shape[1:])
    images = np.moveaxis(images, 2, 1).reshape(H * rows, W * cols, *images.shape[4:])

    if is_torch:
        images = torch.from_numpy(images).permute(2, 0, 1).to(device=device)

    return images


def implay(arr: np.ndarray, fps: float = 30.0, scale: float = 1.0) -> None:
    with NamedTemporaryFile(mode="w", suffix=".mp4") as f:
        # there's a discrepancy between get_writer's interface in v2.pyi
        # and it's definition, so we ignore the type-checking here
        writer = imageio.get_writer(f.name, fps=fps, format="FFMPEG")  # type: ignore
        size = cast(
            tuple[int, int],
            tuple(np.round(np.multiply(arr.shape[1:3], scale)).astype(int)),
        )
        print(f"Writing temporary video file {f.name}")
        for i in tqdm.trange(len(arr)):
            frame = arr[i] if scale == 1.0 else imresize(arr[i], size)
            writer.append_data(frame)
        writer.close()
        display(Video.from_file(f.name))
