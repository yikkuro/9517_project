import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from data_preprocess import edge_detect, shadow_detect, augment_img


image_df = pd.read_csv(
    "dataset/labels.csv", delim_whitespace=True, names=["Filepath", "Label", "Type"]
)

print(image_df.head())

# Shuffle the DataFrame and reset index
image_df = image_df.sample(frac=1).reset_index(drop=True)
image_df["Filepath"] = "dataset/" + image_df["Filepath"].astype(str)

# Show the result
image_df
# Display 15 picture of the dataset with their labels
fig, axes = plt.subplots(
    nrows=3, ncols=5, figsize=(15, 7), subplot_kw={"xticks": [], "yticks": []}
)

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[i]))
    ax.set_title(image_df.Label[i])
plt.tight_layout()
plt.show()
# Separate in train and test data
train_i, test_i = next(
    StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=24).split(
        image_df["Filepath"], image_df["Label"]
    )
)
print(image_df)

train_df = image_df.iloc[train_i]
add_train_df_list = []
for i, row in train_df.iterrows():
    if row["Label"] == 2:
        for _ in range(3):
            add_train_df_list.append(row.copy())
train_df = pd.concat([train_df, pd.DataFrame(add_train_df_list)])
test_df = image_df.iloc[test_i]


def proprocess_func(x):
    # gray_img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # gray_img = augment_img(gray_img, 0.1)
    # x = x.T
    # x[0] = x[1] = x[2] = cv2.resize(
    #     np.concatenate((shadow_detect(gray_img), edge_detect(gray_img))), (224, 224)
    # )
    # return tf.keras.applications.mobilenet_v2.preprocess_input(x.T)
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=proprocess_func,
    validation_split=0.25,
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=proprocess_func
)

# Convert labels to strings
train_df["Label"] = train_df["Label"].astype(str)
test_df["Label"] = test_df["Label"].astype(str)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=50,
    shuffle=True,
    seed=42,
    subset="training",
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
    subset="validation",
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

# Load the pretained model
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
)

pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation="relu")(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)

outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=10,
    # workers=4,
    use_multiprocessing=False,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True
        )
    ],
)

pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[["loss", "val_loss"]].plot()
plt.title("Loss")
plt.show()

results = model.evaluate(test_images, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
