import cv2
import numpy as np
import dataset.utils.elpv_reader as dataset_reader
from sklearn.model_selection import train_test_split
from skimage import feature


def extract_hog_features(image):
    if image is not None:
        hog_feature = feature.hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        return hog_feature


def edge_detect(img):
    rv = img.copy()
    cv2.equalizeHist(rv, rv)
    cv2.medianBlur(rv, 9, rv)
    cv2.Laplacian(rv, -1, rv, 5)
    cv2.medianBlur(rv, 7, rv)
    cv2.threshold(rv, 60, 255, cv2.THRESH_BINARY, rv)
    return rv


def shadow_detect(img):
    rv = img.copy()
    cv2.equalizeHist(rv, rv)
    cv2.medianBlur(rv, 9, rv)
    cv2.threshold(rv, 60, 255, cv2.THRESH_BINARY, rv)
    return cv2.bitwise_not(rv)


def augment_img(img: np.ndarray, chance):
    rv = img.copy()
    if np.random.random() <= chance:
        cv2.flip(rv, 1, rv)
    if np.random.random() <= chance:
        cv2.flip(rv, 0, rv)
    if np.random.random() <= chance:
        brighteness = np.random.randint(-10, 10)
        rv = (rv.astype(np.int32) + brighteness).astype(np.uint8)
    if np.random.random() <= chance:
        guassian_noise = np.random.normal(0, 5, (img.shape[0], img.shape[1])).astype(
            np.int32
        )
        rv = (rv.astype(np.int32) + guassian_noise).astype(np.uint8)
    return rv


def manual_preprocess(img):
    return cv2.resize((shadow_detect(img) + edge_detect(img)), (100, 100))


def elpv_reader_wrapper(type="all", feature_extract_method="manual"):
    images, probs, types = dataset_reader.load_dataset()
    split = 0.25
    possible_probs = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
    label_map = [0, 1, 2, 3]
    labels = [label_map[possible_probs.index(p)] for p in probs]

    if type == "mono" or type == "poly":
        images = [images[i] for i in range(len(images)) if types[i] == type]
        labels = [labels[i] for i in range(len(labels)) if types[i] == type]

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.25, random_state=42, stratify=labels
    )

    augment_chance = 0.5
    duplicate = 3

    new_train_images = []
    new_train_labels = []
    for i in range(len(train_images)):
        if train_labels[i] == 2 or train_labels[i] == 1:
            for _ in range(duplicate):
                new_train_images.append(augment_img(train_images[i], augment_chance))
                new_train_labels.append(train_labels[i])
    train_images = np.concatenate((train_images, np.array(new_train_images)))
    train_labels = np.concatenate((train_labels, np.array(new_train_labels)))

    train_X = []
    test_X = []
    train_Y = train_labels
    test_Y = test_labels

    if feature_extract_method == "manual":
        train_X = [
            [pixel for row in manual_preprocess(image) for pixel in row]
            for image in train_images
        ]
        test_X = [
            [pixel for row in manual_preprocess(image) for pixel in row]
            for image in test_images
        ]

    elif feature_extract_method == "hog":
        train_X = [extract_hog_features(image) for image in train_images]
        test_X = [extract_hog_features(image) for image in test_images]

    return train_X, test_X, train_Y, test_Y
