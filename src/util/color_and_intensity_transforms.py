# Sumegha Singhania and Heather Fryling
# Northeastern University

import cv2
import numpy as np
import random

def min_max_img(img):
  '''
  Get the minimum and maximum values for each color channel in an image.
  img: np array of shape (rows, cols, 3)
  '''
  min_list = []
  max_list = []
  
  # Split image channels
  channels = cv2.split(img)

  # Loop over channels and get min and max values
  for _, channel in enumerate(channels):
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(channel)
      min_list.append(min_val)
      max_list.append(max_val)
  return min_list,max_list

def min_max_coeff(min_list, max_list, img_max=65535):
  '''
  Gets the minimum and maximum coefficient per color channel without causing clipping in the upper ranges.
  min_list: the minimums from min_max_img
  max_list: the maximums from min_max_img
  img_max (optional): value to divide the image by to scale the range to [0, 1].
  '''
  min_coeff = []
  max_coeff = []

  max_coeff = [img_max/x for x in max_list]

  for i in range(3):
    min_val = min_list[i]
    min_coeff_val = min_val/img_max

    if min_val>=4 and min_coeff_val<0.25:
      min_coeff_val = 0.25
    elif min_val<=3 and min_val>2 and min_coeff_val<0.33:
      min_coeff_val = 0.33
    elif min_val<=2 and min_val>1 and min_coeff_val<0.5:
      min_coeff_val = 0.5
    else:
      min_coeff_val = 0.5

    min_coeff.append(min_coeff_val)

  return min_coeff,max_coeff

def apply_random_color_balance(img, img_max_val=65535):
    '''
    Applies random color balance to an image.
    img: np array of shape (rows, cols, 3)
    img_max_val: the maximum value of the image
    '''
    min_list,max_list = min_max_img(img)
    min_coeff,max_coeff = min_max_coeff(min_list,max_list)
    color_coeff = [random.uniform(min_coeff[i], max_coeff[i]) for i in range(3)]
    img[:,:,0]*=color_coeff[0]
    img[:,:,1]*=color_coeff[1]
    img[:,:,2]*=color_coeff[2]

    img = np.clip(img, 0, img_max_val)          
    return img

def apply_random_intensity_variation(img, img_max_val=65535):
  '''
  Applies random intensity variation to an image.
  img: np array of shape (rows, cols, 3)
  img_max_val: the maximum value of the input image
  '''
  min_list,max_list = min_max_img(img)
  min_coeff,max_coeff = min_max_coeff(min_list,max_list)
  min_val = min(min_coeff[0],min_coeff[1],min_coeff[2])
  max_val = max(max_coeff[0],max_coeff[1],max_coeff[2])
  coeff = random.uniform(min_val, max_val)
  img *= coeff
  img = np.clip(img, 0, img_max_val)
  return img

def apply_color_and_intensity_variation(img):
  '''
  Applies random color and intensity variation to an image.
  img: np array of shape (rows, cols, 3)
  '''
  img = apply_random_color_balance(img)
  img = apply_random_intensity_variation(img)
  return img

def apply_color_intensity_transform_random_chance(img):
    '''
    Has a uniform random chance of doing nothing, applying random color variation, 
    applying random intensity variation, or applying random color and random intensity variation to an image.
    img: np array of shape (rows, cols, 3)
    '''
    random_number = random.random()
    if random_number < .25:
      img = apply_random_color_balance(img)
    elif random_number < .5:
      img = apply_random_intensity_variation(img)
    elif random_number < .75:
       img = apply_color_and_intensity_variation(img)
    return img
  