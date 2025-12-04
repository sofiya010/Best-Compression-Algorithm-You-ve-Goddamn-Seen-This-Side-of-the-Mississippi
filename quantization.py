# quantization and dequantization of an 8by 8 DCT block

from typing import List # type hiunts

# standard JPEG-ish luminance quantization matrix (quality ~50)
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

# for simplicity chroma channels use the same table
STANDARD_CHROMA_Q = STANDARD_LUMA_Q

# divide each DCT coefficient by corresponding quantization value + round up to nearest int
# gets rid of high frequency detail and is the main reason jpeg is a lossy compression
def quantize_block(block: List[List[float]], q_matrix: List[List[int]]) -> List[List[int]]:
   n = 8 # jpeg blocks are always 8 by 8
   out = [[0] * n for _ in range(n)] # empty 8 by 8 int block

   for y in range(n): # rows
      for x in range(n): # columns
         out[y][x] = int(round(block[y][x] / q_matrix[y][x]))
   return out

# multiple int coefficients back by the quantization matrix so inverse DCT can reconstruct the img
def dequantize_block(block: List[List[int]], q_matrix: List[List[int]]) -> List[List[float]]:
   n = 8 # you already know, 8 by 8 bayby
   out = [[0.0] * n for _ in range(n)] # output is float again; inverse DCT
   for y in range(n):
      for x in range(n):
         out[y][x] = block[y][x] * q_matrix[y][x]
   return out
