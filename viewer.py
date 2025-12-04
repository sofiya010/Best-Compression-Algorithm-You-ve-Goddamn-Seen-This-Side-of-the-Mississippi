import sys
import struct
import numpy as np
from typing import List

from PIL import Image

from quantization import STANDARD_LUMA_Q, STANDARD_CHROMA_Q, dequantize_block
from twoDDCT import idct_2d, unblockify
from entropyEncoding import rle_decode, inverse_zigzag_scan
from colorConversion import ycbcr_to_rgb_image

file_signature = b"JPCS"
header_version = 2

# read binary .jpc file made by compressor
# validates header, laods image size, block count, and all rle data
def get_image(path: str) -> dict:

   with open(path, "rb") as f:
      # validate final file signature
      file_sig = f.read(4)
      if file_sig != file_signature:
         raise ValueError("Not a JPCS file")

      # check version for compatibility
      version = struct.unpack(">B", f.read(1))[0]
      if version != header_version:
         raise ValueError(f"Unsupported JPCS version: {version}")

      # image dimensions and block size
      width, height = struct.unpack(">II", f.read(8))
      block_size = struct.unpack(">B", f.read(1))[0]

      # number of encoded blocks for each channel
      y_count, cb_count, cr_count = struct.unpack(">III", f.read(12))

      # read all RLE blocks for one channel
      def read_channel(block_count):
         blocks = []
         for _ in range(block_count):
               pair_count = struct.unpack(">H", f.read(2))[0]
               rle_block = []
               for _ in range(pair_count):
                  zeros, value = struct.unpack(">Bh", f.read(3))
                  rle_block.append((zeros, value))
               blocks.append(rle_block)
         return blocks

      y_blocks = read_channel(y_count)
      cb_blocks = read_channel(cb_count)
      cr_blocks = read_channel(cr_count)

   # Return structured dictionary for the decoder
   return {
      "width": width,
      "height": height,
      "block_size": block_size,
      "y_blocks": y_blocks,
      "cb_blocks": cb_blocks,
      "cr_blocks": cr_blocks,
   }

# Reverse the full compression process for a single channel:
def decompress_channel(blocks_rle, q_matrix, width, height, block_size):
   reconstructed_blocks = []

   for rle_block in blocks_rle:
      zz = rle_decode(rle_block, total_length=64)
      q_block = inverse_zigzag_scan(zz, block_size)
      dct_block = dequantize_block(q_block, q_matrix)

      # convert to float array and apply inverse DCT
      arr = np.array(dct_block, dtype=np.float32)
      spatial_block = idct_2d(arr)
      reconstructed_blocks.append(spatial_block)

   # turn list of blocks back into the full channel
   channel = unblockify(reconstructed_blocks, height, width, block_size)
   return channel


def main():
   # require at least a .jpc file path
   if len(sys.argv) < 2:
      print("Usage:")
      print("  python3 viewer.py compressed.jpc [output.png]")
      sys.exit(1)

   jpc_path = sys.argv[1]
   output_path = sys.argv[2] if len(sys.argv) >= 3 else "view_from_jpc.png"

   print("Loading compressed file:", jpc_path)
   compressed = get_image(jpc_path)

   width = compressed["width"]
   height = compressed["height"]
   block_size = compressed["block_size"]

   print("Reconstructing channels...")

   # decode Y, Cb, and Cr channels separately
   y_channel = decompress_channel(
      compressed["y_blocks"], STANDARD_LUMA_Q, width, height, block_size
   )
   cb_channel = decompress_channel(
      compressed["cb_blocks"], STANDARD_CHROMA_Q, width, height, block_size
   )
   cr_channel = decompress_channel(
      compressed["cr_blocks"], STANDARD_CHROMA_Q, width, height, block_size
   )

   print("Converting YCbCr to RGB...")
   img = ycbcr_to_rgb_image(y_channel, cb_channel, cr_channel)

   print("Saving to:", output_path)
   img.save(output_path)

   # display if possible
   try:
      img.show()
   except Exception:
      pass

   print("Done.")


if __name__ == "__main__":
   main()
