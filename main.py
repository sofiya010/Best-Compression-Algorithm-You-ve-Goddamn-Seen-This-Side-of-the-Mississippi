# This is a mini JPEG Compression project focused on 4 steps; Color Conversion, DCT, Quantization, and Entropy Encoding
# This is our main driver file where most of the parser and function calls are created.


# imports
import os
import struct # for packing and unpacking binary data in .jpc files
import numpy as np # numpy; arrays and math
from PIL import Image # pillow image library

# color space functions
from colorConversion import rgb_to_ycbcr_image, ycbcr_to_rgb_image

# entropy encoding functions
from entropyEncoding import zigzag_scan, inverse_zigzag_scan, rle_encode, rle_decode

from twoDDCT import ( # functions from the DCT file
   blockify,
   unblockify,
   dct_2d,
   idct_2d
)

from quantization import ( # functions from quantization
   quantize_block,
   dequantize_block,
   STANDARD_LUMA_Q,
   STANDARD_CHROMA_Q,
)

# JPEG constants
block_size = 8                     # DCT blocks are always 8 by 8
file_signature = b"JPCS"             # file signature at the start of each .jpc file
header_version = 2                 # version bump whenever format changes

# load an image from disk, always convert to RGB
def get_image(path: str) -> Image.Image:
   image = Image.open(path).convert("RGB") # open w pillow and force rgb
   return image

# write compressed data to custom .jpc binary file
def save_compressed(compressed: dict, path: str) -> None:

   # get the metadata and the channel data from the compressed dictionary
   width = compressed["width"]
   height = compressed["height"]
   block_size = compressed["block_size"]

   # lists of rle blocks for y, cb, and cr
   y_blocks = compressed["y_blocks"] 
   cb_blocks = compressed["cb_blocks"]
   cr_blocks = compressed["cr_blocks"]

   # wb = write binary 
   with open(path, "wb") as file:

      # header (signature, version, sizes)
      file.write(file_signature)
      file.write(struct.pack(">B", header_version))
      file.write(struct.pack(">II", width, height))
      file.write(struct.pack(">B", block_size))
      file.write(struct.pack(">III", len(y_blocks), len(cb_blocks), len(cr_blocks)))

      # writes a list of RLE-coded blocks for a single channel
      def write_channel(blocks):
         for rle_block in blocks:
               # write how many (zero,value) pairs this block holds
               pair_amount = len(rle_block)
               file.write(struct.pack(">H", pair_amount))

               # write each RLE pair; 1byte zeros, 2bytes value (signed short)
               for zeros, value in rle_block:
                  file.write(struct.pack(">Bh", zeros, value))

      # write y, cb, cr channel data in this order
      write_channel(y_blocks)
      write_channel(cb_blocks)
      write_channel(cr_blocks)

# read a .jpc file back into a Python dict 
def read_JPC_file(path: str) -> dict:
   with open(path, "rb") as f:
      # validate file signature and the version
      file_sig = f.read(4) #read 4 byte file signature str
      if file_sig != file_signature:
         raise ValueError("Not a JPCS file")

      version = struct.unpack(">B", f.read(1))[0] # read 1 byte for version
      if version != header_version:
         raise ValueError(f"Unsupported version: {version}")

      # basic metadata
      width, height = struct.unpack(">II", f.read(8))
      block_size = struct.unpack(">B", f.read(1))[0]
      y_count, cb_count, cr_count = struct.unpack(">III", f.read(12))

      # read all rle blocks for a channel
      def read_channel(block_count):
         blocks = []
         for _ in range(block_count):
               # read amnt of (zero, value) pairs
               pair_count = struct.unpack(">H", f.read(2))[0]
               rle_block = []
               #read that amount of pairs
               for _ in range(pair_count):
                  zeros, value = struct.unpack(">Bh", f.read(3))
                  rle_block.append((zeros, value))
               blocks.append(rle_block)
         return blocks

      # read all y, cb, cr blocks
      y_blocks = read_channel(y_count)
      cb_blocks = read_channel(cb_count)
      cr_blocks = read_channel(cr_count)

   #return dictionary in the same format as the one made by compress_image
   return {
      "width": width,
      "height": height,
      "block_size": block_size,
      "y_blocks": y_blocks,
      "cb_blocks": cb_blocks,
      "cr_blocks": cr_blocks,
   }

# compress a single Y, Cb, or Cr channel
def compress_channel(channel, q_matrix):
   # channel compression pipeline: Y/Cb/Cr array then 8×8 blocks then DCT then quantize then zigzag then RLE
   
   # make sure we have a float32 NumPy array
   channel = np.asarray(channel, dtype=np.float32)

   # split full 2D channel into a list of 8 by 8 blocks
   blocks = blockify(channel, block_size)
   compressed_blocks = []

   for block in blocks:
      dct_block = dct_2d(block) # frequency transform
      q_block = quantize_block(dct_block, q_matrix)# lossy quantization
      zigzag = zigzag_scan(q_block) # reorder for RLE
      rle = rle_encode(zigzag) # compress zero runs
      compressed_blocks.append(rle) # append encoded blocks

   return compressed_blocks

# compress the entire RGB image (really YCbCr)
def compress_image(img: Image.Image):
   width, height = img.size

   # convert RGB to Y, Cb, Cr (luminance + two chromanance channels)
   y_channel, cb_channel, cr_channel = rgb_to_ycbcr_image(img)

   # compress each channel with its appropriate quantization table
   y_blocks = compress_channel(y_channel, STANDARD_LUMA_Q)
   cb_blocks = compress_channel(cb_channel, STANDARD_CHROMA_Q)
   cr_blocks = compress_channel(cr_channel, STANDARD_CHROMA_Q)

   # metadata and compressed channel data into the dictionary- you know the drill
   return {
      "width": width,
      "height": height,
      "block_size": block_size,
      "y_blocks": y_blocks,
      "cb_blocks": cb_blocks,
      "cr_blocks": cr_blocks,
   }

# reverse channel compression (for viewer)
def decompress_channel(blocks_rle, q_matrix, width, height, block_size):
   reconstructed_blocks = [] # will hold 8 by 8 spcial domain blocks

   for rle_block in blocks_rle:
      zz = rle_decode(rle_block, total_length=64) # undo RLE
      q_block = inverse_zigzag_scan(zz, block_size) # undo zigzag
      dct_block = dequantize_block(q_block, q_matrix) # undo quantization
      arr = np.array(dct_block, dtype=np.float32) # make sure it's float32
      spatial_block = idct_2d(arr) # inverse DCT
      reconstructed_blocks.append(spatial_block) # crtl s tha shi

   # blocks back into 2D image
   return unblockify(reconstructed_blocks, height, width, block_size)

# full decompression (viewer only)
def decompress_image(compressed: dict) -> Image.Image:
   width = compressed["width"]
   height = compressed["height"]
   block_size = compressed["block_size"]

   y_chan = decompress_channel(compressed["y_blocks"], STANDARD_LUMA_Q, width, height, block_size)
   cb_chan = decompress_channel(compressed["cb_blocks"], STANDARD_CHROMA_Q, width, height, block_size)
   cr_chan = decompress_channel(compressed["cr_blocks"], STANDARD_CHROMA_Q, width, height, block_size)

   # convert back to RGB pillow style
   return ycbcr_to_rgb_image(y_chan, cb_chan, cr_chan)

# driver function; main 
def main():
   input_path = "input.bmp" # maybe switch later if time, otherwise whatever. feature not a bug.
   compressed_path = "compressed.jpc"

   img = get_image(input_path) # get input image

   # resize large images so compression is faster for testing
   if img.width > 800 or img.height > 533:
      img = img.resize((800, 533))
   print("Loaded image:", img.size)
   width, height = img.size

   # compress and write result to .jpc
   compressed = compress_image(img)
   print("Compression produced", len(compressed["y_blocks"]), "Y blocks")

   # write compressed representation to .jpc file
   save_compressed(compressed, compressed_path)
   print("Saved compressed file to", compressed_path)

   # size comparisons 
   raw_rgb_bytes = width * height * 3
   original_size = os.path.getsize(input_path)
   compressed_size = os.path.getsize(compressed_path)
   print("\nSize comparison (bytes):")
   print(f"  Raw RGB (width*height*3): {raw_rgb_bytes}")
   print(f"  Original input file:       {original_size}")
   print(f"  Custom JPC file:           {compressed_size}")

   # compression ratios
   raw_ratio = raw_rgb_bytes / compressed_size
   file_ratio = original_size / compressed_size
   print("\nCompression ratios:")
   print(f"  Raw RGB  : Custom JPC ≈ {raw_ratio:.2f}:1")
   print(f"  PNG file : Custom JPC ≈ {file_ratio:.2f}:1")

# run tha shi
if __name__ == "__main__":
   main()
