import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

path = "PNG/"
x = cv2.imread(os.path.join('PNG', 'Original', '2.png'))

print(x)