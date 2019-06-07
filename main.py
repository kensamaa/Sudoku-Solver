
import cv2
import argparse
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import operator
import copy
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from skimage.segmentation import clear_border
from keras.models import load_model

def show_image(img,title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 450,450)
    cv2.imshow(title, img)
    cv2.waitKey(5000)#show the picture for 5 secs
    cv2.destroyAllWindows()

#Image filter processing
def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9),0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
      kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
      proc = cv2.dilate(proc, kernel)
    return proc

img = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
show_image(img,"Original Image")
processed = pre_process_image(img)
show_image(processed,"preProcessed")
print("end")