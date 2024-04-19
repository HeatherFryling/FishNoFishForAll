# Heather Fryling
# 11/2/2022
# Given a source directory and an image directory, 
# this program resizes linear images
# such that the short edge is MIN_EDGE pixels.

import cv2
import os
from tqdm import tqdm

# Directory containing linear tiffs.
SRC_DIR = ''
# Directory for resized linear tiffs.
DST_DIR = ''
TIFF = 'tiff'
MIN_EDGE = 480

files = os.listdir(SRC_DIR)
for i in tqdm(range(len(files)), desc=f'Resizing to short edge of {MIN_EDGE}'):
    item = files[i]
    if item[-4:] == TIFF:
        path = os.path.join(SRC_DIR, item)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        rows = img.shape[0]
        cols = img.shape[1]
        scale_factor = MIN_EDGE / min(rows, cols)
        dim = (int(cols * scale_factor), int(rows * scale_factor))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DST_DIR, f'{item[:-5]}_resized{MIN_EDGE}.tiff'), resized)
print('done')
