from skimage import feature
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier


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


def SVM_pred(train_images, train_labels, test_images):
    train_X = []
    test_X = []
    print("extracting hog...")
    for i in range(len(train_images)):
        train_X.append(extract_hog_features(train_images[i]))
    for i in range(len(test_images)):
        test_X.append(extract_hog_features(test_images[i]))
    print("finish extracting hog")

    bagging_clf = BaggingClassifier(
        estimator=SVC(kernel="rbf"),
        n_estimators=10,
        random_state=0,
        max_features=0.8
    )
    bagging_clf.fit(train_X, train_labels)

    return bagging_clf.predict(test_X)
