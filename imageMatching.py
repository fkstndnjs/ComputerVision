import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import signal as sg
from skimage.feature import hog
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Define the image path
image_path = './recaptcha-dataset/Large/Bicycle/Bicycle (3).png'

# Point processing
# Load BGR image
image = cv2.imread(image_path)

# BGR -> gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR -> HSV
hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Gray -> contrast stretching
stretch = cv2.equalizeHist(gray)

# Area processing - Noise filtering
blur_gauss = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=1)
blur_median = cv2.medianBlur(image, ksize=3)
blur_mean = cv2.blur(image, ksize=(3, 3))

# Edge detection
edge_canny = cv2.Canny(gray, 100, 200)
edge_sobelx = cv2.Sobel(gray, ddepth=-1, dx=1, dy=0, delta=128)
edge_sobely = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1, delta=128)

# Sharpening
sharp = cv2.addWeighted(image, 2, blur_gauss, -1, 0)

# Visualization
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

# Utility function to normalize histogram
def norm_hist(hist):
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

# Color histogram
hist_b, bins_b = np.histogram(image[0], bins=256, range=(0, 256))
hist_g, bins_g = np.histogram(image[1], bins=256, range=(0, 256))
hist_r, bins_r = np.histogram(image[2], bins=256, range=(0, 256))
hist_b = norm_hist(hist_b)    # 256-d
hist_g = norm_hist(hist_g)    # 256-d
hist_r = norm_hist(hist_r)    # 256-d

# Gray histogram
hist_gray, bins_gray = np.histogram(gray, bins=128, range=(0, 256))
hist_gray = norm_hist(hist_gray)    # 128-d

# Visualization
plt.figure(figsize=(20, 3))

plt.subplot(1, 4, 1)
plt.title('blue histogram')
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

# Local Binary Pattern (LBP)
lbp = local_binary_pattern(gray, P=8, R=1)

hist_lbp, bin_lbp = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
hist_lbp = norm_hist(hist_lbp)    # 64-d

# Visualization
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.title('LBP image')
plt.axis('off')
plt.imshow(lbp, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('LBP histogram')
plt.bar(bin_lbp[:-1], hist_lbp, width=1)

# Gray-Level Co-occurrence Matrix (GLCM)
glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)

max_prob = np.max(glcm)
contrast = graycoprops(glcm, 'contrast')
dissimilarity = graycoprops(glcm, 'dissimilarity')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

print('Max probability:', max_prob)
print('Contrast:', contrast[0][0])
print('Dissimilarity:', dissimilarity[0][0])
print('Homogeneity:', homogeneity[0][0])
print('Energy:', energy[0][0])
print('Correlation:', correlation[0][0])

# Law's Texture Energy Measures (TEM)
def laws_texture(gray):
    (rows, cols) = gray.shape[:2]
    
    smooth_kernel = (1/25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray, smooth_kernel, "same")
    gray_processed = np.abs(gray - gray_smooth)
    
    filter_vectors = np.array([[ 1,  4,  6,  4, 1],    # L5
                               [-1, -2,  0,  2, 1],    # E5
                               [-1,  0,  2,  0, 1],    # S5
                               [ 1, -4,  6, -4, 1]])   # R5

    filters = []
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5,1),
                                     filter_vectors[j][:].reshape(1,5)))

    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    texture_maps = []
    texture_maps.append((conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2)     # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2)     # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2)    # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2)    # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2)     # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2)   # S5R5 / R5S5
    texture_maps.append(conv_maps[:, :, 10])                                # S5S5
    texture_maps.append(conv_maps[:, :, 5])                                 # E5E5
    texture_maps.append(conv_maps[:, :, 15])                                # R5R5
    texture_maps.append(conv_maps[:, :, 0])                                 # L5L5 (used to normalize TEM)

    TEM = []
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())
        
    return TEM

laws = laws_texture(gray)    # 9-d
print(laws)

# SIFT feature matching
image1 = cv2.imread('./recaptcha-dataset/Large/Crosswalk/Cross (2).png')
image2 = cv2.imread('./recaptcha-dataset/Large/Crosswalk/Cross (5).png')

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)    # des: (n, 128)
kp2, des2 = sift.detectAndCompute(image2, None)    # des: (n, 128)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
print(len(matches))
matches = sorted(matches, key=lambda x: x.distance)
image3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:], None, flags=2)
plt.imshow(image3)
plt.show()

# Histogram of Oriented Gradients (HoG)
# Extract HoG features
# fd: 8 * (image.shape[0]//16)*(image.shape[1]//16) dimension
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

# Visualize HoG image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap='gray')
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# K-Nearest Neighbors (KNN) classification
recaptcha = './recaptcha-dataset/Large'
labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 
          'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']

train_features = []
train_labels = []
test_features = []
test_labels = []

for label in labels:
    image_dir = os.path.join(recaptcha, label)
    image_list = os.listdir(image_dir)
    for i, image_name in enumerate(image_list):
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if i < 10:
            train_features.append(laws_texture(gray_img))
            train_labels.append(label)
        elif 10 <= i < 20:
            test_features.append(laws_texture(gray_img))
            test_labels.append(label)
        else:
            break
            
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(train_features, train_labels)
predict_labels = classifier.predict(test_features)
print(classification_report(test_labels, predict_labels))
