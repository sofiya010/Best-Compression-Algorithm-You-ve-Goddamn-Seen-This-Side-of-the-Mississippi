# Quantization and dequantization of an 8x8 DCT block.

from typing import List

# Standard JPEG-ish luminance quantization matrix (quality ~50)
STANDARD_LUMA_Q: List[List[int]] = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
]

# For simplicity chroma channels use the same table
STANDARD_CHROMA_Q = STANDARD_LUMA_Q

def quantize_block(block: List[List[float]], q_matrix: List[List[int]]) -> List[List[int]]:
    """
    Divide each DCT coefficient by the corresponding quantization value and round to the nearest integer.
    This gets rid of high-frequency detail and is the primary source of lossy compression.

    """
    n = 8
    out = [[0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            out[y][x] = int(round(block[y][x] / q_matrix[y][x]))
    return out


def dequantize_block(block: List[List[int]], q_matrix: List[List[int]]) -> List[List[float]]:
    """
    Reverse quantization: multiply integer coefficients back by the 
    quantization matrix so the inverse DCT can reconstruct the image
    """
    n = 8
    out = [[0.0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            out[y][x] = block[y][x] * q_matrix[y][x]
    return out

