import math
import librosa
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.model_selection import train_test_split

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
