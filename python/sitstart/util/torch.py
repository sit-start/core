import torch
from PIL import ImageFont, ImageDraw
from torchvision.transforms import PILToTensor, ToPILImage

DEFAULT_FONT_NAME = "/System/Library/Fonts/Monaco.ttf"


def int_hist(
    data: torch.Tensor | list, min: int | None = None, max: int | None = None
) -> torch.Tensor:
    data = data if isinstance(data, torch.Tensor) else torch.Tensor(data)
    data = data.round()
    data = data.to(dtype=torch.float64) if not torch.is_floating_point(data) else data
    min = int(min or data.min().item())
    max = int(max or data.max().item())
    hist = torch.histogram(data, range=(min - 0.5, max + 0.5), bins=max - min + 1)
    return hist.hist.to(dtype=torch.long)


def unnormalize(
    images: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    images = images[(None,) * (4 - images.ndim)]
    result = images * torch.tensor(std).reshape((1, 3, 1, 1)) + torch.tensor(
        mean
    ).reshape(1, 3, 1, 1)
    return result.squeeze(0)


def overlay_text(
    images: torch.Tensor,
    text: str | list[str],
    position: tuple[int, int] = (0, 0),
    font_name: str = DEFAULT_FONT_NAME,
    font_size: int = 10,
    fill: str | tuple[int, int, int] = "yellow",
):
    images = images[(None,) * (4 - images.ndim)]
    if isinstance(text, str):
        text = [text] * images.shape[0]
    if len(text) != images.shape[0]:
        raise ValueError(
            f"Text length ({len(text)}) must match number of images ({images.shape[0]})"
        )

    result = []
    for i, image in enumerate(images):
        font = ImageFont.truetype(font=font_name, size=font_size)
        pil_image = ToPILImage()(image)
        text_overlay = ImageDraw.Draw(pil_image)
        text_overlay.text(position, text[i], fill=fill, font=font)
        result.append(PILToTensor()(pil_image))

    return torch.stack(result).squeeze(0)


def randint(
    *,
    low: int | None = None,
    high: int | None = None,
    size: tuple[int, ...] | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Generate random integers.

    Same as torch.randint with default values for `low`, `high`, and `dtype`.
    """
    if not is_integer(dtype):
        raise ValueError(f"`dtype` must be an integer type, got {dtype!r}.")
    return torch.randint(
        low=low if low is not None else torch.iinfo(dtype).min,
        high=high if high is not None else torch.iinfo(dtype).max + 1,
        size=size or (1,),
        generator=generator,
        dtype=dtype,
    )


def generator_from_seed(
    seed: int | None = None, device: torch.device | str = "cpu"
) -> torch.Generator:
    """Create a torch.Generator from the given seed.

    Uses a random seed if `seed` is None.
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    return generator


def is_integer(input: torch.Tensor | torch.dtype) -> bool:
    """Check if the input is an integer type."""
    dtype = input if isinstance(input, torch.dtype) else input.dtype
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
