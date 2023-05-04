# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:28:44 2023

@author: anadjj
"""

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

from imshowtools import imshow

from helper import create_dir

def load_data(path, split=0.1):
    ## 80 - 10 - 10

    # list of images and masks
    images = sorted(glob(os.path.join(path, "Original/*")))
    masks = sorted(glob(os.path.join(path, "groundTruth/*")))
    
    path2 = "dataset2"
    imagesK = sorted(glob(os.path.join(path2, "images/*")))
    masksK = sorted(glob(os.path.join(path2, "masks/*")))
    
    # for i, image_name in enumerate(imagesK):
    #     image = ImageOps.pad(image_animal, (256,256))
    
    all_images = images + imagesK
    all_masks = masks + masksK
    
    print(f"Images: {len(all_images)} - Masks: {len(all_masks)}")
    
    counter = 0
    
    # visualize images and masks
    for x, y in zip(all_images, all_masks):
        print(x, y)
        counter += 1
        if counter == 10:
            break
    
    cat = []
    for x, y in zip(all_images[0:6], all_masks[:6]):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        
        z = np.concatenate([x, y], axis=1)
        cat.append(z)
    
    imshow(*cat, size=(20, 10), columns=3)
    
    total_size = len(all_images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    #Dataset split
    
    train_x, valid_x = train_test_split(all_images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(all_masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")
    
    print(test_x)
    
    #Creating the folders
    save_dir = os.path.join("dataset", "aug")
    for item in ["train", "valid", "test"]:
        create_dir(os.path.join(save_dir, item, "images"))
        create_dir(os.path.join(save_dir, item, "masks"))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0 #normalization, normalization vs standardization
    ## (256, 256, 3)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    ## (256, 256)
    x = np.expand_dims(x, axis=-1)
    ## (256, 256, 1)
    return x

#building dataset pipeline
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #print(tf_parse(x, y))
    #z, y = tf_parse
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


# if __name__ == "__main__":
#     print("")
#     path = "PNG"
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
#     print(len(train_x), len(valid_x), len(test_x))
    
#     ds = tf_dataset(test_x, test_y)
#     for x, y in ds:
#         print(x.shape, y.shape)
#         break
    









