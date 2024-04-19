# Heather Fryling
# 11/2/2022
# Given a source directory and a destination directory, 
# this program converts raw images to linear tiffs and stores
# the tiffs in the destination directory.

import rawpy
import imageio
import math
import cv2
import os
from tqdm import tqdm

# Directory containing RAW images.
SRC_DIR = './data/raw'
# Directory for linear TIFFs.
DST_DIR = './data/linear'
CR2 = '.CR2'
DNG = '.DNG'
auto_bright_thr = .001


files = os.listdir(SRC_DIR)
for i in tqdm(range(len(files)), desc=f'Linearizing directory.'):
    item = files[i]
    if item[-4:] == CR2 or item[-4:] == DNG:
        path = os.path.join(SRC_DIR, item)
        with rawpy.imread(path) as raw_file:
            num_bits = int(math.log(raw_file.white_level + 1, 2))
            rgb = raw_file.postprocess( gamma=(1,1), no_auto_bright=False, auto_bright_thr= auto_bright_thr, output_bps=16, use_camera_wb=True)
    imageio.imsave(os.path.join(DST_DIR, f'{item[:-4]}_{num_bits}bitcamera.tiff'), rgb)
print('done')

