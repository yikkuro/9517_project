import cv2
import numpy as np
import dataset.utils.elpv_reader as dataset_reader


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


def preprocess(img):
    print(img)
    return np.concatenate((shadow_detect(img), edge_detect(img)))


def elpv_reader_wrapper(type="all"):
    images, probs, types = dataset_reader.load_dataset()
    split = 0.25
    possible_probs = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
    label_map = [0, 1, 2, 3]
    labels = [label_map[possible_probs.index(p)] for p in probs]

    test_mask = [False for _ in range(len(labels))]
    for label in label_map:
        new_mask = [False for _ in range(len(labels))]
        for i in range(len(labels)):
            if labels[i] == label and np.random.random() < split:
                new_mask[i] = True
        test_mask = [test_mask[i] or new_mask[i] for i in range(len(labels))]

    images = [cv2.resize(image, (150, 150)) for image in images]

    if type == "mono" or type == "poly":
        images = [images[i] for i in range(len(images)) if types[i] == type]
        labels = [labels[i] for i in range(len(labels)) if types[i] == type]

    train_images = [images[i] for i in range(len(images)) if not test_mask[i]]
    test_images = [images[i] for i in range(len(images)) if test_mask[i]]

    train_labels = [labels[i] for i in range(len(labels)) if not test_mask[i]]
    test_labels = [labels[i] for i in range(len(labels)) if test_mask[i]]

    augment_chance = 0.5
    duplicate = 3

    new_train_images = []
    new_train_labels = []
    for i in range(len(train_images)):
        if train_labels[i] == 2 or train_labels[i] == 1:
            for _ in range(duplicate):
                new_train_images.append(augment_img(train_images[i], augment_chance))
                new_train_labels.append(train_labels[i])
    train_images.extend(new_train_images)
    train_labels.extend(new_train_labels)

    for i in range(len(train_images)):
        train_images[i] = preprocess(train_images[i])

    for i in range(len(test_images)):
        test_images[i] = preprocess(test_images[i])

    return train_images, test_images, train_labels, test_labels
