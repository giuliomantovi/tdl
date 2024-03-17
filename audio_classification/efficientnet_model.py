import os
import numpy as np
import tensorflow as tf
import cv2
import sys
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Config import Constants
from mfcc_models import plot_hist

genre_dict = {0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "HipHop", 5: "Jazz",
              6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"}


def load_resize_image_test(img_folder):
    x = []
    names = []
    for root, subdirs, files in os.walk(img_folder):
        for filename in files:
            x = x + [cv2.resize(cv2.imread(os.path.join(root, filename)), (200, 200))]
            names.append(filename)
    return np.array(x), names


def load_resize_image_data(img_folder):
    x = []
    y = []
    for genre_folder in os.listdir(img_folder):
        x = x + [cv2.resize(cv2.imread(os.path.join(img_folder, genre_folder, curr_file)), (200, 200))
                 for curr_file in os.listdir(os.path.join(img_folder, genre_folder))]
        y = y + [genre_folder] * len(os.listdir(os.path.join(img_folder, genre_folder)))
    return np.array(x), np.array(y)


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )


def create_pretrained_efficientnet_model(image_folder):
    x_img, y_img = load_resize_image_data(image_folder)
    label_encoder = LabelEncoder()
    y_img = label_encoder.fit_transform(y_img)

    x_train, x_test, y_train, y_test = train_test_split(x_img, y_img, test_size=0.22, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Each image in the dataset is of the shape (288, 432, 3).
    input_shape = x_img.shape[1:]
    inputs = layers.Input(shape=input_shape)
    print(input_shape)
    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        # classifier_activation='softmax',
    )

    model.trainable = False
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(units=Constants.NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    epochs = 50  # @param {type: "slider", min:8, max:80}
    """ epochs frozen = 50, epochs unfrozen = 20 batch_size=64
    loss: 0.1789 - accuracy: 0.9342 - val_loss: 1.2579 - val_accuracy: 0.7051"""
    hist = model.fit(x_train, y_train,
                     epochs=epochs,  # 100
                     validation_data=(x_val, y_val),
                     batch_size=64)
    # plot_hist(hist)

    # 2 step
    unfreeze_model(model)
    epochs = 20
    hist = model.fit(x_train, y_train,
                     epochs=epochs,  # 100
                     validation_data=(x_val, y_val),
                     batch_size=64)
    plot_hist(hist)

    model.save("audio_classification/GTZAN_DB/models/GTZAN_EFFICIENTNETB0.h5")
    y_pred = model.predict(x_img)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)


def testefficientnetmodel(images_path, model_path):
    x_img, names = load_resize_image_test(images_path)
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_img)
    translate_predictions(y_pred, names)
    # y_pred = np.argmax(y_pred, axis=1)


def translate_predictions(predictions, names):
    for i in range(len(predictions)):
        print(names[i].split(".")[0])
        three_largest, indexes = find3largest(predictions[i])
        tot = sum(three_largest)
        for j in range(len(three_largest)):
            three_largest[j] = round(three_largest[j] / tot * 100, 0)
        for j in range(len(three_largest)):
            if three_largest[j] != 0.:
                print(genre_dict[indexes[j]] + ": " + str(int(three_largest[j])) + "%")
        print()


def find3largest(array, arr_size=Constants.NUM_CLASSES):
    # time complexity o(n), space complexity o(1)
    if arr_size < 3:
        print(" Invalid Input ")
        return
    third = first = second = -sys.maxsize
    third_index = second_index = first_index = -1

    for i in range(0, arr_size):
        if array[i] > first:
            third = second
            third_index = second_index
            second = first
            second_index = first_index
            first = array[i]
            first_index = i
        elif array[i] > second:
            third = second
            third_index = second_index
            second = array[i]
            second_index = i
        elif array[i] > third:
            third = array[i]
            third_index = i

    return [first, second, third], [first_index, second_index, third_index]


"""def create_scratch_efficientnet_model(image_folder):
    x_img, y_img = load_resize_image_data(image_folder)
    label_encoder = LabelEncoder()
    y_img = label_encoder.fit_transform(y_img)

    x_train, x_test, y_train, y_test = train_test_split(x_img, y_img, test_size=0.22, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Each image in the dataset is of the shape (288, 432, 3).
    input_shape = x_img.shape[1:]
    print(input_shape)
    model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True,
        weights=None,
        input_shape=input_shape,
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    epochs = 100
    hist = model.fit(x_train, y_train,
                     epochs=epochs,  # 100
                     validation_data=(x_val, y_val))
    plot_hist(hist)

    model.save("GTZAN_DB/GTZAN_SCRATCH_EFFICIENTNETB0.h5")
    y_pred = model.predict(x_img)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)"""