import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from os import listdir
from matplotlib.image import imread
from google.colab import drive

drive.mount("/content/drive")
plt.figure(figsize=(12, 12))
path = "/content/drive/kongkea/species_dataset/species_1"

for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(path + '/' + random.choice(sorted(listdir(path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize=10)
    plt.ylabel(rand_img.shape[0], fontsize=10)

dir = "/content/drive/kongkea/species_dataset"
root_dir = listdir(dir)
image_list, label_list = [], []
for directory in root_dir:
    for files in listdir(f"{dir}/{directory}"):
        image_path = f"{dir}/{directory}/{files}"
        image = cv2.imread(image_path)
        image = img_to_array(image)
        image_list.append(image)
        label_list.append(directory)
label_counts = pd.DataFrame(label_list).value_counts()
label_counts
num_classes = len(label_counts)
num_classes
image_list[0].shape
label_list = np.array(label_list)
label_list.shape
x_train, x_test, y_train, y_test = train_test_split(
    image_list, label_list, test_size=0.2, random_state=10
)
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype=np.float16) / 225.0
x_train = x_train.reshape(-1, 224, 224, 3)
x_test = x_test.reshape(-1, 224, 224, 3)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print(lb.classes_)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2)
model = Sequential()
model.add(
    Conv2D(8, (3, 3), padding="same", input_shape=(
        224, 224, 3), activation="relu")
)
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(0.0005), metrics=["accuracy"]
)
epochs = 50
batch_size = 128
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
)
model.save("/content/drive/kongkea/species.h5")
plt.figure(figsize=(12, 5))
plt.plot(history.history["accuracy"], color="r")
plt.plot(history.history["val_accuracy"], color="b")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train", "val"])
plt.show()
plt.figure(figsize=(12, 5))
plt.plot(history.history["loss"], color="r")
plt.plot(history.history["val_loss"], color="b")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["train", "val"])
plt.show()
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
y_pred = model.predict(x_test)
img = array_to_img(x_test[5])
img
labels = lb.classes_
print(labels)
print("Originally : ", labels[np.argmax(y_test[5])])
print("Predicted : ", labels[np.argmax(y_pred[5])])
