import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

# visualization
plt.figure(figsize=(8, 8))

plt.subplot(3, 3, 1)
plt.title('image')
plt.axis('off')
plt.imshow(image)

plt.subplot(3, 3, 2)
plt.title('gray')
plt.axis('off')
plt.imshow(gray, cmap='gray')

plt.subplot(3, 3, 3)
plt.title('HSI')
plt.axis('off')
plt.imshow(hsi)

plt.subplot(3, 3, 4)
plt.title('contrast stretch')
plt.axis('off')
plt.imshow(stretch, cmap='gray')

plt.subplot(3, 3, 5)
plt.title('gaussian filter')
plt.axis('off')
plt.imshow(blur_gauss)

plt.subplot(3, 3, 6)
plt.title('median filter')
plt.axis('off')
plt.imshow(blur_median)

plt.subplot(3, 3, 7)
plt.title('canny edge')
plt.axis('off')
plt.imshow(edge_canny, cmap='gray')

plt.subplot(3, 3, 8)
plt.title('sobel edge')
plt.axis('off')
plt.imshow(edge_sobelx, cmap='gray')

plt.subplot(3, 3, 9)
plt.title('sharpening')
plt.axis('off')
plt.imshow(sharp)

def norm_hist(hist):
    # Normalize the histogram
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

# color histogram
hist_b, bins_b = np.histogram(image[0], bins=256, range=(0, 256))
hist_g, bins_g = np.histogram(image[1], bins=256, range=(0, 256))
hist_r, bins_r = np.histogram(image[2], bins=256, range=(0, 256))
hist_b = norm_hist(hist_b)    # 256-d
hist_g = norm_hist(hist_g)    # 256-d
hist_r = norm_hist(hist_r)    # 256-d

# gray histogram
hist_gray, bins_gray = np.histogram(gray, bins=128, range=(0, 256))
hist_gray = norm_hist(hist_gray)    # 128-d

# visualization
plt.figure(figsize=(20, 3))

plt.subplot(1, 4, 1)
plt.title('blue histogran')
plt.bar(bins_b[:-1], hist_b, width=1)

plt.subplot(1, 4, 2)
plt.title('green histogram')
plt.bar(bins_g[:-1], hist_g, width=1)

plt.subplot(1, 4, 3)
plt.title('red histogram')
plt.bar(bins_r[:-1], hist_r, width=1)

plt.subplot(1, 4, 4)
plt.title('gray histogram')
plt.bar(bins_gray[:-1], hist_gray, width=1)

from skimage.feature import local_binary_pattern

# LBP
lbp = local_binary_pattern(gray, P=8, R=1)

hist_lbp, bin_lbp = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
hist_lbp = norm_hist(hist_lbp)    # 64-d

# visualization
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.title('LBP image')
plt.axis('off')
plt.imshow(lbp, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('LBP histogram')
plt.bar(bin_lbp[:-1], hist_lbp, width=1)