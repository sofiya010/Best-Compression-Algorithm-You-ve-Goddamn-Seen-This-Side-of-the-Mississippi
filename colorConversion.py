# Step 1; Color Conversion
# We are creating a Chrominance/Luminance Color Space fromt he RGB color space passed in by main

# imports
from typing import List, Tuple # type hints for lists and xyz tuples
from PIL import Image # Pillow's image class 
import numpy as np # numbers and math

# makes sure a number is in the 0 to 255 range, so final image valies are valid for rgb / YCbCr
def clamp(value: float, min_val: int = 0, max_val: int = 255) -> int:
   return max(min_val, min(max_val, int(round(value))))

# convert 1 rgb pixel to YCbCr using BT.601 equations; standard transform used by JPEG; output is clamped
def rgb_to_ycbcr_pixel(r: int, g: int, b: int) -> Tuple[int, int, int]:
   # weighted sum for luminance (Y); weighted sum of r, g, and b (weights based off human sensitivity to the color)
   y = 0.299 * r + 0.587 * g + 0.114 * b

   # blue-difference and red-difference chroma components; negative from r, positive for b, then offset w 128
   cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
   cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

   return clamp(y), clamp(cb), clamp(cr) # c l a m p

# inverse bt.601 transformations; YCbCr pixel back to RGB by undoing the shifts then converting back
def ycbcr_to_rgb_pixel(y: int, cb: int, cr: int) -> Tuple[int, int, int]:
   
   # remve the 128 offset to get signed chroma
   cb_shift = cb - 128
   cr_shift = cr - 128

   # reconstruct R, G, B using standard inverse coefficients
   r = y + 1.402 * cr_shift
   g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
   b = y + 1.772 * cb_shift

   return clamp(r), clamp(g), clamp(b) # c l a m p (part 2)

# convert pillow rgb image to 3 separate 2d numpy arrays
def rgb_to_ycbcr_image(img: Image.Image):
   width, height = img.size
   pixels = img.load()  # direct pixel access for speed

   # make Python lists first; convert to NumPy later
   y_channel  = [[0] * width for _ in range(height)]
   cb_channel = [[0] * width for _ in range(height)]
   cr_channel = [[0] * width for _ in range(height)]

   # convert each pixel independently
   for j in range(height): # j is row index
      for i in range(width): # i is column index
         r, g, b = pixels[i, j] # read rgb tuple from the original img

         # convert pixel from rgb to ycbcr
         y_val, cb_val, cr_val = rgb_to_ycbcr_pixel(r, g, b)

         # call me grocery the way I store tha shi
         y_channel[j][i]  = y_val
         cb_channel[j][i] = cb_val
         cr_channel[j][i] = cr_val

   # convert lists to float32 NumPy arrays for further processing (DCT, etc.)
   return (
      np.array(y_channel, dtype=np.float32),
      np.array(cb_channel, dtype=np.float32),
      np.array(cr_channel, dtype=np.float32),
   )

# convert the 3 2d Y Cb Cr arrays to pillow rgb 
def ycbcr_to_rgb_image(y_channel, cb_channel, cr_channel) -> Image.Image:

   # make sure inputs are numpy, convert if not
   y_channel = np.array(y_channel)
   cb_channel = np.array(cb_channel)
   cr_channel = np.array(cr_channel)

   # get image dimensions from y channel
   height, width = y_channel.shape

   # create a blank RGB image w same width/height
   img = Image.new("RGB", (width, height))
   pixels = img.load()

   # convert each pixel from YCbCr to RGB
   for j in range(height): # row
      for i in range(width): # col
         y_val  = y_channel[j][i]
         cb_val = cb_channel[j][i]
         cr_val = cr_channel[j][i]

         # convert ycbcr pixel to an rgb triple
         r, g, b = ycbcr_to_rgb_pixel(y_val, cb_val, cr_val)
         pixels[i, j] = (r, g, b) # write rgb pixel into pillow

   return img
