import os
import cv2
import numpy as np
import csv
from scipy import signal as sg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def extract_texture_features(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)  # Resize the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    # Law's texture
    (rows, cols) = gray_img.shape[:2]

    smooth_kernel = (1 / 25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray_img, smooth_kernel, "same")
    gray_processed = np.abs(gray_img - gray_smooth)

    filter_vectors = np.array([[1, 4, 6, 4, 1],  # L5
                               [-1, -2, 0, 2, 1],  # E5
                               [-1, 0, 2, 0, 1],  # S5
                               [1, -4, 6, -4, 1]])  # R5

    # 0:L5L5, 1:L5E5, 2:L5S5, 3:L5R5,
    # 4:E5L5, 5:E5E5, 6:E5S5, 7:E5R5,
    # 8:S5L5, 9:S5E5, 10:S5S5, 11:S5R5,
    # 12:R5L5, 13:R5E5, 14:R5S5, 15:R5R5
    filters = list()
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5, 1),
                                     filter_vectors[j][:].reshape(1, 5)))

    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2)  # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2)  # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2)  # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2)  # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2)  # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2)  # S5R5 / R5S5
    texture_maps.append(conv_maps[:, :, 10])  # S5S5
    texture_maps.append(conv_maps[:, :, 5])  # E5E5
    texture_maps.append(conv_maps[:, :, 15])  # R5R5
    texture_maps.append(conv_maps[:, :, 0])  # L5L5 (use to norm TEM)

    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())

    return TEM

labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
          'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']

recaptcha_train = './recaptcha-dataset/Large'
recaptcha_test = './query'

train_features = []
train_labels = []
test_features = []
test_labels = []

# Train dataset collection
for label in labels:
    image_dir = os.path.join(recaptcha_train, label)
    image_list = os.listdir(image_dir)
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        features = extract_texture_features(image_path)

        train_features.append(features)
        train_labels.append(label)

# Test dataset collection
image_dir = os.path.join(recaptcha_test)
image_list = os.listdir(image_dir)
for image_name in image_list:
    image_path = os.path.join(image_dir, image_name)
    features = extract_texture_features(image_path)

    test_features.append(features)
    test_labels.append(label)

# Classifier training and evaluation
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_features, train_labels)
predict_labels = classifier.predict(test_features)
# print(classification_report(test_labels, predict_labels))

predict_labels = classifier.predict(test_features)
print(predict_labels)

with open('c2_t1_a1.csv','w') as file :
    write = csv.writer(file)
    for i, predict_label in enumerate(predict_labels):
        write.writerow([f'query{i+1}.png', predict_label])

neigh_ind = classifier.kneighbors(X=test_features, n_neighbors=10, return_distance=False) # Top-10 results
neigh_labels = np.array(train_labels)[neigh_ind]  

with open('c2_t2_a1.csv','w') as file :
    write = csv.writer(file)
    for i, neigh_label in enumerate(neigh_labels):
        write.writerow([f'query{i+1}.png'] + list(neigh_label))