from data_preprocess import elpv_reader_wrapper
from SVM_classifier import SVM_pred
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time

start = time.time()
print("loading images...")
train_images, test_images, train_labels, test_labels = elpv_reader_wrapper(
    type="all", feature_extract_method="manual"
)
print("finish loading images")


print("SVM fitting & predicting...")
pred_labels = SVM_pred(train_images, train_labels, test_images)
print("finish SVM")
end = time.time()
print(f"time used: {end-start} ms")


print(accuracy_score(test_labels, pred_labels))
print(confusion_matrix(test_labels, pred_labels))
