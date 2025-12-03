from typing import List, Tuple


# Zigzag order for an 8x8 block
ZIGZAG_INDEXES = [
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


def zigzag_scan(block: List[List[int]]) -> List[int]:
    """
    Convert 8x8 block to 1D vector using zigzag order.
    """
    return [block[y][x] for (y, x) in ZIGZAG_INDEXES]


def inverse_zigzag_scan(vec: List[int], n: int = 8) -> List[List[int]]:
    """
    Convert 1D zigzag vector back to 8x8 block.
    """
    block = [[0] * n for _ in range(n)]
    for idx, (y, x) in enumerate(ZIGZAG_INDEXES):
        block[y][x] = vec[idx]
    return block


def rle_encode(coeffs: List[int]) -> List[Tuple[int, int]]:
    """
    Run-length encode a list of integers with lots of trailing zeros.
    Returns list of (run_length_of_zeros, next_nonzero_value).
    We stop at the last non-zero to avoid encoding infinite zeros.
    """
    # Trim trailing zeros
    last_nonzero = -1
    for i, v in enumerate(coeffs):
        if v != 0:
            last_nonzero = i
    if last_nonzero == -1:
        # Entire block zero
        return [(0, 0)]

    trimmed = coeffs[: last_nonzero + 1]

    result: List[Tuple[int, int]] = []
    zero_count = 0

    for v in trimmed:
        if v == 0:
            zero_count += 1
        else:
            result.append((zero_count, v))
            zero_count = 0

    # If last entries were zeros, they'd be trimmed by last_nonzero logic above
    return result


def rle_decode(encoded: List[Tuple[int, int]], total_length: int = 64) -> List[int]:
    """
    Decode RLE created by rle_encode back into a list of length total_length (64).
    """
    coeffs = []
    for zeros, value in encoded:
        coeffs.extend([0] * zeros)
        coeffs.append(value)

    # Pad with zeros to full length
    if len(coeffs) < total_length:
        coeffs.extend([0] * (total_length - len(coeffs)))
    else:
        coeffs = coeffs[:total_length]

    return coeffs
