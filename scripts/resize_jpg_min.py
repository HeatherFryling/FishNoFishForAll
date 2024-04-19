# Heather Fryling
# 11/2/2022
# Given a source directory and an destination directory, 
# this program resizes all jpgs in the source directory
# such that the minimum edge is of size MIN_EDGE.

import cv2
import os
from tqdm import tqdm

# Directory containing JPEG images.
SRC_DIR = './data/jpg'
# Directory for resized JPEG images.
DST_DIR = './data/jpg_resized'
JPG = '.JPG'
JPEG = '.jpeg'
MIN_EDGE = 64

files = os.listdir(SRC_DIR)

for i in tqdm(range(len(files)), desc=f'Resizing jpg to short edge of {MIN_EDGE}'):
    item = files[i]
    if item[-4:] == JPG or item[-5:] == JPEG:
        path = os.path.join(SRC_DIR, item)
        img = cv2.imread(path)
        rows = img.shape[0]
        cols = img.shape[1]
        scale_factor = MIN_EDGE / min(rows, cols)
        dim = (int(cols * scale_factor), int(rows * scale_factor))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DST_DIR, f'{item[:-4]}_resized{MIN_EDGE}.jpg'), resized)
print('done')