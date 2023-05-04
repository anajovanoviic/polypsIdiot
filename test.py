import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

path = "PNG/"
images = sorted(glob(os.path.join(path, "Original/*")))
masks = sorted(glob(os.path.join(path, "groundTruth/*")))

