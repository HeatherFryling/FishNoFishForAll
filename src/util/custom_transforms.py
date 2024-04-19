# Heather Fryling
# Northeastern University

import random
import numpy as np


def random_square_crop(img):
  ''' 
  Expects a 3D numpy array of shape (rows, cols, channels). Returns a random square crop with
      sides the length of the short side of the image.
  Use this instead of torchvision transforms to avoid the PIL 8-bit bottleneck.
  '''
  rows, cols, channels = img.shape
  if rows < cols:
    left = random.randint(0, cols - rows)
    return img[:, left:left + rows]
  if cols < rows:
    top = random.randint(0, rows - cols)
    return img[top:top + cols, :]
  
def center_square_crop(img):
  ''' 
  Expects a 3D numpy array of shape (rows, cols, channels). Returns a center square crop with
      sides the length of the short side of the image.
  Use this instead of torchvision transforms to avoid the PIL 8-bit bottleneck.
  '''
  rows, cols, _ = img.shape
  if rows < cols:
    left = 0 + (cols - rows) // 2
    return img[:, left:left + rows]
  if cols < rows:
    top = 0 + (rows - cols) // 2
    return img[top:top + cols, :]
  

def linear_to_srgb(img, input_max_value=65535, format_max_value=255):
  '''
  Transforms an image from linear to sRGB.
  img: a numpy array
  input_max_value (optional): maximum value of the input image
  format_max_value (optional): maximum value of the format converting to
  In the default case, the function expects an input image as a 16-bit integer and simulates
  moving to an 8-bit sRGB image, then normalizes the output to [0, 1].
  '''
  img = img / input_max_value
  img = rgb_to_srgb(img)
  img *= format_max_value
  img = img.astype(int)
  img = img.astype(float)
  img /= format_max_value
  return img

def linear_to_float_linear(img, input_max_value=65535):
  '''
  Normalizes a linear image to [0, 1].
  img: a numpy array
  input_max_value (optional): maximum value of the input image
  '''
  img = img / input_max_value
  return img

def linear_to_log(img):
  '''
  Takes the log of a linear image.
  img: a numpy array
  '''
  img_min = np.min(img)
  img_max = np.max(img)
  img = img.astype("float32")
  img[img!=0] = np.log(img[img!=0]) # Do not take log of 0.
  return img

def rgb_to_srgb(linear_img):
  '''
  Takes a linear image scaled range [0, 1].
  Outputs the same image in sRGB in range [0, 1].
  linear_img: a numpy array
  '''
  low_mask = linear_img <= 0.0031308
  high_mask = linear_img > 0.0031308
  linear_img[low_mask] *= 12.92
  linear_img[high_mask] = ((linear_img[high_mask]*1.055)**(1/2.4)) - 0.055
  linear_img[linear_img > 1.0] = 1.0
  linear_img[linear_img < 0.0] = 0
  return linear_img

def srgb_to_plinear(srgb_img, input_max_val=255):
  '''
  Convert an srgb image to linear.
  srgb_img: a numpy array
  input_max_val (optional): the maximum value of the input image
  '''
  lin_img = srgb_img / input_max_val
  if np.max(lin_img) > 1 or np.min(lin_img) < 0:
      raise Exception("srgb_img must be scaled to range [0, 1]. Use max_val to automatically scale the image.")
  low_mask = lin_img <= 0.04045
  high_mask = lin_img > 0.04045
  lin_img[low_mask] /= 12.92
  lin_img[high_mask] = (((srgb_img[high_mask]+ 0.055)/1.055)**(2.4))
  lin_img[srgb_img > 1.0] = 1.0
  lin_img[srgb_img< 0.0] = 0
  return lin_img

def plinear_to_log(linear_img, max_val=65535):
  '''
  Converts a linear or pseudo-linear image scaled to range [0, 1] to log or pseudo-log.
  linear_img: a numpy array
  input_max_val (optional): the maximum value to take the log of
  '''
  log_img = linear_img.copy()
  log_img *= max_val
  return linear_to_log(log_img)

