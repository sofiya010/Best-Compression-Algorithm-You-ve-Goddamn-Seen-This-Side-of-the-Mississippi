from typing import List, Tuple
from PIL import Image
import numpy as np


def _clamp(value: float, min_val: int = 0, max_val: int = 255) -> int:
    """
    Clamp a numeric value to the valid 0–255 byte range.
    Ensures final image values stay valid for RGB/Y/Cb/Cr.
    """
    return max(min_val, min(max_val, int(round(value))))


def rgb_to_ycbcr_pixel(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """
    Convert one RGB pixel → YCbCr using the BT.601 equations.
    This is the standard transform used by JPEG.
    Output components are clamped to 0–255.
    """
    # Weighted sum for luminance (Y)
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # Blue-difference and red-difference chroma components
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    return _clamp(y), _clamp(cb), _clamp(cr)


def ycbcr_to_rgb_pixel(y: int, cb: int, cr: int) -> Tuple[int, int, int]:
    """
    Inverse BT.601 transform: convert one YCbCr pixel -> RGB.
    Undo the shifts (Cb/Cr stored with +128 offset) and convert back.
    """
    cb_shift = cb - 128
    cr_shift = cr - 128

    # Reconstruct R, G, B using standard inverse coefficients
    r = y + 1.402 * cr_shift
    g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
    b = y + 1.772 * cb_shift

    return _clamp(r), _clamp(g), _clamp(b)


def rgb_to_ycbcr_image(img: Image.Image):
    """
    Convert a full Pillow RGB image → three separate 2D NumPy arrays:
        Y channel, Cb channel, Cr channel.
    Each array has shape (height, width).
    """
    width, height = img.size
    pixels = img.load()  # Direct pixel access for speed

    # Allocate Python lists first; convert to NumPy later
    y_channel  = [[0] * width for _ in range(height)]
    cb_channel = [[0] * width for _ in range(height)]
    cr_channel = [[0] * width for _ in range(height)]

    # Convert each pixel independently
    for j in range(height):
        for i in range(width):
            r, g, b = pixels[i, j]
            y_val, cb_val, cr_val = rgb_to_ycbcr_pixel(r, g, b)

            y_channel[j][i]  = y_val
            cb_channel[j][i] = cb_val
            cr_channel[j][i] = cr_val

    # Convert lists → float32 NumPy arrays for further processing (DCT, etc.)
    return (
        np.array(y_channel, dtype=np.float32),
        np.array(cb_channel, dtype=np.float32),
        np.array(cr_channel, dtype=np.float32),
    )


def ycbcr_to_rgb_image(y_channel, cb_channel, cr_channel) -> Image.Image:
    """
    Convert three 2D Y/Cb/Cr arrays -> a Pillow RGB image.
    Assumes channels are NumPy arrays of shape (height, width).
    """
    y_channel = np.array(y_channel)
    cb_channel = np.array(cb_channel)
    cr_channel = np.array(cr_channel)

    height, width = y_channel.shape

    # Create a blank RGB image
    img = Image.new("RGB", (width, height))
    pixels = img.load()

    # Convert each pixel from YCbCr → RGB
    for j in range(height):
        for i in range(width):
            y_val  = y_channel[j][i]
            cb_val = cb_channel[j][i]
            cr_val = cr_channel[j][i]

            r, g, b = ycbcr_to_rgb_pixel(y_val, cb_val, cr_val)
            pixels[i, j] = (r, g, b)

    return img
