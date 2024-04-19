# Heather Fryling
# 11/2/2022
# Given a source directory and an image directory, 
# this program converts raw images to linear tiffs and
# resizes them such that the short edge is MIN_EDGE pixels.

import rawpy
import imageio
import math
import cv2
import os
from tqdm import tqdm

# Directory containing RAW images.
SRC_DIR = './data/raw'
# Directory for resized linear TIFFs.
DST_DIR = './data/linear'
CR2 = '.CR2'
DNG = '.DNG'
MIN_EDGE = 64

files = os.listdir(SRC_DIR)
for i in tqdm(range(len(files)), desc=f'Linearizing and resizing to short edge of {MIN_EDGE}'):
    item = files[i]
    if item[-4:] == CR2 or item[-4:] == DNG:
        path = os.path.join(SRC_DIR, item)
        with rawpy.imread(path) as raw_file:
            num_bits = int(math.log(raw_file.white_level + 1, 2))
            rgb = raw_file.postprocess( gamma=(1,1), no_auto_bright=False, auto_bright_thr= .001, output_bps=16, use_camera_wb=True)
            rows = rgb.shape[0]
            cols = rgb.shape[1]
            scale_factor = MIN_EDGE / min(rows, cols)
            dim = (int(cols * scale_factor), int(rows * scale_factor))
            resized = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
    imageio.imsave(os.path.join(DST_DIR, f'{item[:-4]}_{num_bits}bitcamera_resized{MIN_EDGE}.tiff'), resized)
print('done')
