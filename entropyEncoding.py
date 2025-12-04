
from typing import List, Tuple # type hints


# zigzag order for an 8 by 8 block
zigzag_indexes = [
   (0, 0),
   (0, 1), (1, 0),
   (2, 0), (1, 1), (0, 2),
   (0, 3), (1, 2), (2, 1), (3, 0),
   (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
   (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
   (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
   (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
   (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
   (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
   (7, 3), (6, 4), (5, 5), (4, 6), (3, 7),
   (4, 7), (5, 6), (6, 5), (7, 4),
   (7, 5), (6, 6), (5, 7),
   (6, 7), (7, 6),
   (7, 7),
]

# convert 8 by 8 block into 1d vector using zigzag order
def zigzag_scan(block: List[List[int]]) -> List[int]:
   # for each row / col in the indexes, take element from block and convert to 1d list of 64 coefficients
   return [block[y][x] for (y, x) in zigzag_indexes]

# zigzag backwards; convert 1d zigzag vector back to 8 by 8 blocks
def inverse_zigzag_scan(vec: List[int], n: int = 8) -> List[List[int]]:
   block = [[0] * n for _ in range(n)] # n by n zero block
   for idx, (y, x) in enumerate(zigzag_indexes): # go through zz coordinate list and palce each value back
      block[y][x] = vec[idx]
   return block

# rle encode a list of ints with a crap ton of trailing 0s. Will return list of rle zeroes and the next non0 value
# end at last non0 so that we dont have infinite zeroes; pretty sure that would tank our space
def rle_encode(coeffs: List[int]) -> List[Tuple[int, int]]:
   # trim trailing zeros
   last_nonzero = -1
   for i, coefficientValue in enumerate(coeffs):
      if coefficientValue != 0:
         last_nonzero = i
   if last_nonzero == -1: # if that stayed at -1, then the whole block is 0
      return [(0, 0)]

   trimmed = coeffs[: last_nonzero + 1] # slice lsit to only keep up to the last non0 value

   result: List[Tuple[int, int]] = [] # will hold zero count, value pairs
   zero_count = 0

   for coefficientValue in trimmed:
      if coefficientValue == 0: # if current value is 0, bump the zero counter
         zero_count += 1
      else: # if non zero; emit zero_count, coefficient value and reset zero count 
         result.append((zero_count, coefficientValue))
         zero_count = 0

   # if last entries were zeros they should be trimmed by last_nonzero logic above
   return result

# decode rle created by above back into a list of lenfth = total length = 64
def rle_decode(encoded: List[Tuple[int, int]], total_length: int = 64) -> List[int]:
   coeffs = [] # will be full list of coefficients
   for zeros, value in encoded: 
      coeffs.extend([0] * zeros) # add 0s
      coeffs.append(value) # add non 0 value

   # pad with zeros to full length if we dont yet have total_length vales
   if len(coeffs) < total_length:
      coeffs.extend([0] * (total_length - len(coeffs)))
   else: # if it's somehow longer (with some kind of divine intervention), truncate it
      coeffs = coeffs[:total_length]

   return coeffs
