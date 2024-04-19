# Heather Fryling
# 11/2/2022
# Given a source directory and an image directory, 
# this program converts images to log images
# in exr 32-bit floating point format.

# image io requires an additional binary to
# deal with exr images.
# Find it here: https://imageio.readthedocs.io/en/v2.8.0/format_exr-fi.html.

import imageio
import cv2
import os
import numpy as np
from tqdm import tqdm

# Directory containing linear TIFFs.
SRC_DIR = './data/linear'
# Directory for log EXRs.
DST_DIR = './data/log'
TIFF = '.tiff'
files = os.listdir(SRC_DIR)


for i in tqdm(range(len(files)), desc=f'Converting to log.'):
    item = files[i]
    if item[-5:] == TIFF:
        path = os.path.join(SRC_DIR, item)
        log_image = cv2.imread(path, cv2.IMREAD_UNCHANGED) #BGR
        log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB) # convert to RGB before save
        log_image = log_image.astype("float32")
        log_image[log_image!=0] = np.log(log_image[log_image!=0]) # Do not take log of 0.
        imageio.imsave(os.path.join(DST_DIR, f'{item[:-5]}.exr'), log_image) 
print('done')