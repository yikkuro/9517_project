import cv2
import numpy as np
import dataset.utils.elpv_reader as dataset_reader

images, probs, types = dataset_reader.load_dataset()

possible_probs = [1.0, 0.6666666666666666, 0.3333333333333333, 0.0]
possible_types = ["mono", "poly"]


def print_data_size(type: str, prob: float):
    print(
        f"{type} {prob}:",
        len(
            [
                probs[i]
                for i in range(len(probs))
                if types[i] == type and probs[i] == prob
            ]
        ),
    )


print("data size:", len(probs))
for type in possible_types:
    print("--------")
    for prob in possible_probs:
        print_data_size(type, prob)


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


def elpv_reader_wrapper():
    new_images = []
    for image in images:
        augmented_new_image = augment_img(image, 0.1)
        new_images.append(
            np.concatenate(
                (edge_detect(augmented_new_image), shadow_detect(augmented_new_image)),
                axis=1,
            )
        )
    return new_images, probs, types


new_images, new_probs, new_types = elpv_reader_wrapper()

cv2.imshow("test", new_images[46])
cv2.waitKey(0)
