import math
import json
import librosa
import os
import numpy as np
import tensorflow as tf
import cv2
import sys
import matplotlib.pyplot as plt
from keras import layers, models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Config import Constants


def preprocess_dataset(dataset_path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10):
    data = {"labels": [], "mfcc": [], "duration": [], "filenames": []}
    sample_rate = 22050
    samples_per_segment = int(sample_rate * 30 / num_segment)
    for label_idx, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath == dataset_path:
            continue
        print(label_idx)
        for f in sorted(filenames):
            if not f.endswith('.wav'):
                continue
            file_path = str(str(dirpath).split('\\')[-1]) + "/" + str(f)
            # print("Track Name ", file_path)
            try:
                y, sr = librosa.load(dataset_path + "/" + file_path, sr=sample_rate)
            except:
                continue
            data["duration"].append(librosa.get_duration(y=y, sr=sr))
            data["filenames"].append(f)
            for n in range(num_segment):
                mfcc = librosa.feature.mfcc(y=y[samples_per_segment * n: samples_per_segment * (n + 1)], sr=sample_rate,
                                            n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                if len(mfcc) == math.ceil(samples_per_segment / hop_length):
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(label_idx - 1)
    return data


def createLSMTmodel(mfcc_data):
    x = np.array(mfcc_data["mfcc"])
    y = np.array(mfcc_data["labels"])
    print(list(x))
    print(list(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(units=Constants.NUM_CLASSES, activation="softmax"))

    optimiser = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=60, verbose=2)
    model.save("GTZAN/GTZAN_LSTM_2.h5")
    # testing accuracy
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(np.sum(y_pred == y_test) / len(y_pred))


# rete neurale convoluzionale, input = mfcc degli audio, accuracy:90%
def createCNNmodel(mfcc_data):
    x = np.array(mfcc_data["mfcc"])
    y = np.array(mfcc_data["labels"])

    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    y_train[y_train == 10] = 9
    y_val[y_val == 10] = 9
    y_test[y_test == 10] = 9

    input_shape = x_train.shape[1:]

    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=input_shape),
        layers.MaxPooling2D(2, padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid'), layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid'), layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'), layers.Dense(units=Constants.NUM_CLASSES, activation='softmax')
    ])
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
    #N.B. BINARY CROSS ENTROPY???
    cnn_model.summary()

    history = cnn_model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50, verbose=2)
    cnn_model.save("GTZAN/GTZAN_CNN.h5")
    # testing accuracy
    y_pred = cnn_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(np.sum(y_pred == y_test) / len(y_pred))


# for creating the model with GTZAN dataset
def load_image_data(img_folder):
    x = []
    y = []
    for genre_folder in os.listdir(img_folder):
        x = x + [cv2.imread(os.path.join(img_folder, genre_folder, curr_file))
                 for curr_file in os.listdir(os.path.join(img_folder, genre_folder))]
        y = y + [genre_folder] * len(os.listdir(os.path.join(img_folder, genre_folder)))
    return np.array(x), np.array(y)


def load_resize_image_data(img_folder):
    x = []
    y = []
    for genre_folder in os.listdir(img_folder):
        x = x + [cv2.resize(cv2.imread(os.path.join(img_folder, genre_folder, curr_file)), (200, 200))
                 for curr_file in os.listdir(os.path.join(img_folder, genre_folder))]
        y = y + [genre_folder] * len(os.listdir(os.path.join(img_folder, genre_folder)))
    return np.array(x), np.array(y)


# for predicting input from user
def load_image_test(img_folder):
    x = []
    for root, subdirs, files in os.walk(img_folder):
        for filename in files:
            x = x + [cv2.imread(os.path.join(root, filename))]
    return np.array(x)


def load_resize_image_test(img_folder):
    x = []
    names = []
    for root, subdirs, files in os.walk(img_folder):
        for filename in files:
            x = x + [cv2.resize(cv2.imread(os.path.join(root, filename)), (200, 200))]
            names.append(filename)
    return np.array(x), names


# rete neurale convoluzionale, input = spettrogrammi in png degli audio, accuracy:62%
def createCNNimagemodel(image_folder):
    x_img, y_img = load_image_data(image_folder)
    label_encoder = LabelEncoder()
    y_img = label_encoder.fit_transform(y_img)

    x_train, x_test, y_train, y_test = train_test_split(x_img, y_img, test_size=0.22, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Each image in the dataset is of the shape (288, 432, 3).
    input_shape = x_train.shape[1:]

    image_model = None
    image_model = models.Sequential()
    image_model.add(layers.Conv2D(128, 3, padding="same", activation="relu", input_shape=input_shape))
    image_model.add(layers.MaxPool2D())
    image_model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    image_model.add(layers.MaxPool2D())
    image_model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    image_model.add(layers.MaxPool2D())
    image_model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    image_model.add(layers.MaxPool2D())
    image_model.add(layers.Dropout(0.2))
    image_model.add(layers.Flatten())
    image_model.add(layers.Dense(128, activation="relu"))
    image_model.add(layers.Dense(units=Constants.NUM_CLASSES, activation="softmax"))
    image_model.summary()

    opt = Adam(learning_rate=0.0001)
    image_model.compile(optimizer=opt,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    history = image_model.fit(x_train, y_train,
                              epochs=50,  # 100
                              validation_data=(x_val, y_val),
                              batch_size=16,  # 32
                              verbose=2)
    image_model.save("GTZAN/GTZAN_IMAGE_CNN.h5")
    # test
    y_pred = image_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    print(np.sum(y_pred == y_test) / len(y_pred))


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

    model.save("GTZAN/models/GTZAN_EFFICIENTNETB0.h5")
    y_pred = model.predict(x_img)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)


def create_scratch_efficientnet_model(image_folder):
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

    model.save("GTZAN/GTZAN_SCRATCH_EFFICIENTNETB0.h5")
    y_pred = model.predict(x_img)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def default_testmodel(mfcc_data, model_type):
    x = np.array(mfcc_data["mfcc"])
    y = np.array(mfcc_data["labels"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    model = tf.keras.models.load_model(model_type)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(np.sum(y_pred == y_test) / len(y_pred))


def testaudiomodel(mfcc_data, model_path):
    x = np.array(mfcc_data["mfcc"])
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    print(mfcc_data["duration"])
    numbers_per_audio = []
    for second in mfcc_data["duration"]:
        numbers_per_audio.append(min(math.floor(10 * second / 30), 10))
    for i in range(len(numbers_per_audio)):
        if i != 0:
            numbers_per_audio[i] += numbers_per_audio[i - 1]
    # print(np.sum(y_pred == y_test) / len(y_pred))
    print(mfcc_data["filenames"])
    y_pred = np.split(y_pred, numbers_per_audio[:-1])
    print(y_pred)
    audio_genres = []
    for arr in y_pred:
        audio_genres.append(most_frequent(arr))
    print(audio_genres)
    return audio_genres


def testimagemodel(images_path, model_path):
    x_img = load_image_test(images_path)
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_img)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)


def testefficientnetmodel(images_path, model_path):
    x_img, names = load_resize_image_test(images_path)
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_img)
    translate_predictions(y_pred, names)
    # y_pred = np.argmax(y_pred, axis=1)


genre_dict = {0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "HipHop", 5: "Jazz",
              6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"}


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


def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    index = np.argmax(counts)
    return unique[index]


"""
def plot_history(history, name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.plot(history["acc"], label="Training")
    plt.plot(history["val_acc"], label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.savefig(os.path.join("data/plots", "{}_accuracy_{}.png".format(name, timestamp)))

    plt.gcf().clear()

    plt.plot(history["loss"], label="Training")
    plt.plot(history["val_loss"], label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.savefig(os.path.join("data/plots", "{}_loss_{}.png".format(name, timestamp)))


def plot_confusion_matrix(cm, classes, cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)


def save_confusion_matrix(y_test, y_pred, name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=genre_dict.keys())
    plt.tight_layout()
    plt.savefig(os.path.join("data/plots", "{}_confusion_{}.png".format(name, timestamp)))
    """
