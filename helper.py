# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:20:33 2023

@author: anadjj
"""

import os
from tqdm import tqdm
import cv2
from albumentations import HorizontalFlip, CoarseDropout, RandomBrightness, RandomContrast

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def save_dataset(images, masks, save_dir, augment=False):
    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("\\")[-1].split(".")[0]
        
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        
        if augment == True:
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            aug = CoarseDropout(p=1, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = RandomBrightness(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']
            
            aug = RandomContrast(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']
            
            aug_x = [x, x1, x2, x3, x4]
            aug_y = [y, y1, y2, y3, y4]
            
        else:
            aug_x = [x]
            aug_y = [y]
            
        idx = 0
        for ax, ay in zip(aug_x, aug_y):
            aug_name = f"{name}_{idx}.png"
            
            save_image_path = os.path.join(save_dir, "images", aug_name)
            save_mask_path = os.path.join(save_dir, "masks", aug_name)
            
            print(save_image_path)
            
            #n = cv2.imwrite(os.path.join(save_dir, "images", aug_name), ax)
            n = cv2.imwrite(os.path.join(save_dir, "images", aug_name), ax)
            print(n)
            
            cv2.imwrite(save_mask_path, ay)
            
            idx += 1
            
            
        
        