
# color_conversion.py
#
# RGB <-> YCbCr conversion at the per-pixel level and for entire Pillow images.

from typing import List, Tuple
from PIL import Image
import numpy as np


def _clamp(value: float, min_val: int = 0, max_val: int = 255) -> int:
    return max(min_val, min(max_val, int(round(value))))


def rgb_to_ycbcr_pixel(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """
    ITU-R BT.601 conversion (approximate, standard for JPEG-like operations).
    Output ranges roughly [0,255].
    """
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    return _clamp(y), _clamp(cb), _clamp(cr)


def ycbcr_to_rgb_pixel(y: int, cb: int, cr: int) -> Tuple[int, int, int]:
    """
    Inverse BT.601.
    """
    cb_shift = cb - 128
    cr_shift = cr - 128

    r = y + 1.402 * cr_shift
    g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
    b = y + 1.772 * cb_shift

    return _clamp(r), _clamp(g), _clamp(b)


def rgb_to_ycbcr_image(img: Image.Image):
    """
    Convert a Pillow RGB image to three 2D NumPy arrays: Y, Cb, Cr.
    """
    width, height = img.size
    pixels = img.load()

    y_channel: List[List[int]] = [[0] * width for _ in range(height)]
    cb_channel: List[List[int]] = [[0] * width for _ in range(height)]
    cr_channel: List[List[int]] = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            yy, cb, cr = rgb_to_ycbcr_pixel(r, g, b)
            y_channel[y][x] = yy
            cb_channel[y][x] = cb
            cr_channel[y][x] = cr

    return (
        np.array(y_channel, dtype=np.float32),
        np.array(cb_channel, dtype=np.float32), 
        np.array(cr_channel, dtype=np.float32), 

    )

def ycbcr_to_rgb_image(
    y_channel,
    cb_channel,
    cr_channel,
) -> Image.Image:
    """
    Convert 2D Y/Cb/Cr arrays back to a Pillow RGB image.
    """
    y_channel  = np.array(y_channel)
    cb_channel = np.array(cb_channel)
    cr_channel = np.array(cr_channel)

    height, width = y_channel.shape

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for j in range(height):
        for i in range(width):
            y_val = y_channel[j][i]
            cb_val = cb_channel[j][i]
            cr_val = cr_channel[j][i]
            r, g, b = ycbcr_to_rgb_pixel(y_val, cb_val, cr_val)
            pixels[i, j] = (r, g, b)

    return img
