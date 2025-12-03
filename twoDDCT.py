import numpy as np
import math

# splits a 2D channel into 8×8 blocks
def blockify(channel, block_size=8):
    h, w = channel.shape
    blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            
            # extract an 8x8 region
            block = channel[i:i+block_size, j:j+block_size]

            #  if the block is smaller, pad with zeros
            if block.shape != (block_size, block_size):
                block = np.pad(
                    block,
                    (
                        (0, block_size - block.shape[0]),
                        (0, block_size - block.shape[1])
                    ),
                    mode='constant'
                )

            blocks.append(block.astype(np.float32))

    return blocks


# merges blocks back into 2D channels
def unblockify(blocks, height, width, block_size=8):
    # compute padded dimensions
    padded_height = ((height + block_size - 1) // block_size) * block_size
    padded_width  = ((width  + block_size - 1) // block_size) * block_size

    rebuilt = np.zeros((padded_height, padded_width), dtype=np.float32)
    idx = 0

    # place each block back to its original position
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            rebuilt[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    # remove padded edges if needed
    return rebuilt[:height, :width]


# normalization constant
def _alpha(u):
    return 1 / math.sqrt(2) if u == 0 else 1


# 2D discrete cosine transofrom on 8x8 blocks of float32
def dct_2d(block):
    N = block.shape[0]
    result = np.zeros_like(block)

    for u in range(N):
        for v in range(N):
            sum_val = 0.0

            # sum over all spatial coordinates
            for x in range(N):
                for y in range(N):
                    sum_val += block[x, y] * math.cos((2*x+1)*u*math.pi/(2*N)) * math.cos((2*y+1)*v*math.pi/(2*N))

            result[u, v] = 0.25 * _alpha(u) * _alpha(v) * sum_val

    return result


# inverse of dct function
def idct_2d(block):
    N = block.shape[0]
    result = np.zeros_like(block)

    for x in range(N):
        for y in range(N):
            sum_val = 0.0

            # sum over all frequency coordinates
            for u in range(N):
                for v in range(N):
                    sum_val += (
                        _alpha(u)
                        * _alpha(v)
                        * block[u, v]
                        * math.cos((2*x+1)*u*math.pi/(2*N))
                        * math.cos((2*y+1)*v*math.pi/(2*N))
                    )

            result[x, y] = 0.25 * sum_val

    return result
