#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:11:20 2026

@author: mhurtubise
"""

import cv2
import numpy as np
import os

def convert_to_mnist(image_path):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Image not found or path is incorrect")
    
    # 2. Invert if necessary (MNIST: digit is white on black background)
    if np.mean(img) > 127:  # mostly white background
        img = 255 - img
    
    # 3. Threshold to make sure digit is clear
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Find bounding box of the digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]
    
    # 5. Resize while keeping aspect ratio
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
    
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 6. Pad to 28x28
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    
    mnist_img = cv2.copyMakeBorder(digit_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return mnist_img

def convert_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):  # only images
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                mnist_img = convert_to_mnist(input_path)
                cv2.imwrite(output_path, mnist_img)
                print(f"Converted {filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert folder of images to MNIST format")
    parser.add_argument("input_folder", type=str, help="Path to input folder with original images")
    parser.add_argument("output_folder", type=str, help="Path to output folder for MNIST-style images")
    
    args = parser.parse_args()
    
    # Call your function
    convert_folder(args.input_folder, args.output_folder)
            
            
            
            