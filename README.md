# FishNoFishRepo

This is the example code for the FishNoFish dataset for comparison of JPEG, linear TIFF, and log EXR images. This project explores differential performance in a small CNN given the same data in sRGB JPEG, linear TIFF, and log EXR format.

## Paper: Log RGB Images Provide Invariance to Intensity and Color Balance Variation for Convolutional Networks
https://papers.bmvc2023.org/0635.pdf

## Thesis: Rethinking Image Formats for Computer Vision: JPEG sRGB, Linear, and Log RGB; Object Detection and Shadow Removal
https://www.proquest.com/docview/3103720780/abstract/DD503B6CBDAF4B92PQ/1?sourcetype=Dissertations%20&%20Theses

## Data
The FishNoFish dataset consists of 1592 triplets sRGB JPEG, linear RGB TIFF, and log RGB EXR images split into 1273 training images, 158 validation images, and 161 test images. The images are resized to short side of 64 pixels.

The camera originals are also provided for those who wish to perform their own experiments with RAW images.

Get the data here:
https://drive.google.com/drive/u/0/folders/1QDGZKRlRRUT8QHTOaY_6I7vMyQbpg-95

## Installing the Project
1. Clone this repo to your local machine.
2. Download the [data](https://drive.google.com/drive/u/0/folders/1QDGZKRlRRUT8QHTOaY_6I7vMyQbpg-95).
3. Install Python 3.10.
4. Move to the project root directory.
5. Install requirements with `pip install -r requirements.txt`

## Jupyter Notebooks
The Jupyter notebooks can be used to recreate the FishNoFish experiments.

### JPG64.ipynb
This notebook trains FishNet64 on JPEG sRGB images.

### Linear64.ipynb
This notebook trains FishNet64 on 16-bit linear RGB TIFF images.

### Log64.ipynb
This notebook trains FishNet64 on 32-bit log RGB EXR images.

### DynamicSRGB64.ipynb
This notebook trains FishNet64 on sRGB images with random color and intensity variation.

### Dynamic Linear64.ipynb
This notebook trains FishNet64 on 16-bit linear RGB TIFF images with random color and intensity variation.

### Log64.ipynb
This notebook trains FishNet64 on 32-bit log RGB EXR images with random color and intensity variation.

## Image Processing Scripts

### Usage
Example scripts are provided for post-processing RAW images and converting to log RGB.

To move from RAW to linear, use `linearize.py`.

To move from RAW to linear and specify the minimum edge size, use `linearize_resize_min.py`.

To move from RAW to log, first use `linearize.py` or `linearize_resize_min.py`, then `convert_to_log.py`. Never resize a log image. Do all resizing in linear RGB to prevent interpolation errors.

To resize JPEG files, use `resize_jpg_min.py`.

### convert_to_log.py

Given a folder containing 16-bit tiff files and a destination folder, this program
converts all images in the source folder to 32-bit exr log images and saves them
in the destination folder.

### linearize.py
Given a source directory containing .DNG or .CR2 files, linearizes them and stores them in
a destination directory.

### linearize_resize_min.py

Given a source directory and a destination directory. This script converts all raw files
in the source directory to 16-bit tiffs with minimum edge of size MIN_EDGE.

### resize_jpg_min.py

Given a source directory of jpg images and a destination directory, resizes all
images in the source directory and saves them in the destination directory.

### resize_linear_min.py

Given a source directory and a destination directory, resizes all images in the source directory and saves them to the destination directory.
