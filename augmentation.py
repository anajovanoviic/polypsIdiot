# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 03:18:37 2023

@author: anadjj
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, CoarseDropout, RandomBrightness, RandomContrast

from data0Combine import load_data

from helper import save_dataset

path = "PNG"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

save_dir = os.path.join("dataset", "aug")

save_dataset(train_x, train_y, os.path.join(save_dir, "train"), augment=True)
save_dataset(valid_x, valid_y, os.path.join(save_dir, "valid"), augment=False)

save_dataset(test_x, test_y, os.path.join(save_dir, "test"), augment=False)




