import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

print('test')

image_path = './recaptcha-dataset/Large/Bicycle/Bicycle (3).png'

# point processing
# load BGR image
image = cv2.imread(image_path)

# BGR -> gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR -> HSV
hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# gray -> contrast stretching
stretch = cv2.equalizeHist(gray)

# area processing
# noise filtering
blur_gauss = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=1)
blur_median = cv2.medianBlur(image, ksize=3)
blur_mean = cv2.blur(image, ksize=(3, 3))

# edge
edge_canny = cv2.Canny(gray, 100, 200)
edge_sobelx = cv2.Sobel(gray, ddepth=-1, dx=1, dy=0, delta=128)
edge_sobely = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1, delta=128)

# sharpening
sharp = cv2.addWeighted(image, 2, blur_gauss, -1, 0)