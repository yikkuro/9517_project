from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier


def SVM_pred(train_images, train_labels, test_images):
    bagging_clf = BaggingClassifier(
        estimator=SVC(kernel="rbf"),
        n_estimators=10,
        random_state=0,
        max_features=0.8,
        max_samples=0.8,
    )
    bagging_clf.fit(
        train_images,
        train_labels,
    )

    return bagging_clf.predict(test_images)
