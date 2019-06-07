
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
    # Invert colours, so gridlines have non-zero pixel values.
	# Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
      kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
      proc = cv2.dilate(proc, kernel)
    return proc


#Find image corners
def findCorners(img):
    # Find the contours in the image
    # cv2.RETR_TREE indicates how the contours will be retrieved:
    # See: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    #
    # cv2.CHAIN_APPROX_SIMPLE condenses the contour information, only storing the minimum number of points to describe
    # the contour shape.
    h,contours, hierarchy = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

	# Bottom-right point has the largest (x + y) value
	# Top-left has point smallest (x + y) value
	# Bottom-left point has smallest (x - y) value
	# Top-right point has largest (x - y) value

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

#Function used to specify point
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    show_image(img,"display_points")
    return img



img = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
show_image(img,"Original Image")
processed = pre_process_image(img)
show_image(processed,"preProcessed")

corners = findCorners(processed)
display_points(processed, corners)
print("end")
