# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:10:39 2023

@author: anadjj
"""

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageOps

image = Image.open("PNG/Original/1.png")
image.save('1.png')
print(image.size)
print(image.filename)
print(image.format)
print(image.format_description)

imageK = Image.open('dataset2/images/cju0qkwl35piu0993l0dewei2.jpg')
imageK.show()
imageK.save('1K.png')
imageKK = Image.open('1K.png')
print(imageKK.size)
print(imageKK.filename)
print(imageKK.format)
print(imageKK.format_description)

##################

image_animal = Image.open('picture.jpg')

#image_aspect = ImageOps.contain(image_animal, (2048,2048))
#image_aspect = image_animal.thumbnail((256,256), Image.LANCZOS)

image_aspect = ImageOps.pad(image_animal, (256,256))
image_aspect.show()


image_resize = image_animal.resize((256,256))
image_resize.show()

# # better example
scale_factor = 2
new_image_size = (image_animal.size[0] * scale_factor,image_animal.size[1] * scale_factor)
image_resize_better = image_animal.resize(new_image_size)

image_resize_better.show()

image_resize_better.save('scaled.jpg')