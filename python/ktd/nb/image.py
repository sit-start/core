import os
from tempfile import NamedTemporaryFile
from typing import cast
from IPython.display import display

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
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


# adapted https://github.com/znah/notebooks/blob/master/external_colab_snippets.ipynb
def imtile(arr: np.ndarray, cols: int | None = None):
    N, H, W, _ = arr.shape
    arr = np.asarray(arr)
    if cols is None:
        cols = int(np.ceil(np.sqrt(N)))
    H, W = arr.shape[1:3]
    pad = (cols - N) % cols
    arr = np.pad(arr, np.array([(0, pad)] + [(0, 0)] * (arr.ndim - 1)), "constant")
    rows = len(arr) // cols
    arr = arr.reshape(rows, cols, *arr.shape[1:])
    arr = np.moveaxis(arr, 2, 1).reshape(H * rows, W * cols, *arr.shape[4:])
    return arr


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
