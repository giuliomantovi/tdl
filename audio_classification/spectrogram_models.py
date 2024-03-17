import os
import numpy as np
import tensorflow as tf
import cv2
from keras import layers, models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import matplotlib.pyplot as plt

from Config import Constants


def load_image_data(img_folder):
    x = []
    y = []
    for genre_folder in os.listdir(img_folder):
        x = x + [cv2.imread(os.path.join(img_folder, genre_folder, curr_file))
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



def createCNNimagemodel(image_folder):
    # rete neurale convoluzionale, input = spettrogrammi in png degli audio, accuracy:62%
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



def create_mel_spectrogram(audio_path, filename):
    # spectrogram with musical components (that humans can hear)
    y, sr = librosa.load(audio_path)
    s = librosa.feature.melspectrogram(y=y, sr=44100, hop_length=308, win_length=2205,
                                       n_mels=128, n_fft=4096, fmax=18000, norm='slaney')
    s_db_mel = librosa.amplitude_to_db(s, ref=np.max)
    print(s_db_mel.shape)
    # fig, ax = plt.subplots(figsize=(4.32, 2.88))
    fig, ax = plt.subplots(figsize=(2, 2))
    img = librosa.display.specshow(s_db_mel, ax=ax)
    plt.savefig(fname=Constants.INPUT_IMAGES + "/" + filename + ".png", format='png')
    # plt.show()


def audio_to_spectrograms(dir_path):
    for root, subdirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(root, filename)
                create_mel_spectrogram(file_path, filename[:-4])


def testimagemodel(images_path, model_path):
    x_img = load_image_test(images_path)
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_img)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)


"""def create_spectrogram(audio):
    y, sr = librosa.load(audio)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    img = librosa.display.specshow(S_db, ax=ax)
    plt.show()"""