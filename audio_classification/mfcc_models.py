import math
import librosa
import os
import numpy as np
import tensorflow as tf
from audio_classification import general
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


def preprocess_dir(dirpath, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10):
    data = {"labels": [], "mfcc": [], "duration": [], "filenames": []}
    sample_rate = 22050
    samples_per_segment = int(sample_rate * 30 / num_segment)
    label = 0
    print(dirpath)
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(root, file)
            try:
                y, sr = librosa.load(file_path, sr=sample_rate)
            except:
                continue
            data["duration"].append(librosa.get_duration(y=y, sr=sr))
            data["filenames"].append(file)
            for n in range(num_segment):
                mfcc = librosa.feature.mfcc(y=y[samples_per_segment * n: samples_per_segment * (n + 1)], sr=sample_rate,
                                            n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                if len(mfcc) == math.ceil(samples_per_segment / hop_length):
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(label)
                    label += 1
    return data

def createLSMTmodel(mfcc_data):
    x = np.array(mfcc_data["mfcc"])
    y = np.array(mfcc_data["labels"])
    #print(list(x))
    #print(list(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(units=Constants.NUM_CLASSES, activation="softmax"))
    #NO DROPOUT, 50 EPOCHS, 32 bs (LSMT.h5) 188/188 - 3s - loss: 0.0243 - accuracy: 0.9933 - val_loss: 0.8696 - val_accuracy: 0.8318 - 3s/epoch - 15ms/step
    #0.1 dropout, 60 epochs, 32 bs  188/188 - 2s - loss: 0.0292 - accuracy: 0.9912 - val_loss: 0.9628 - val_accuracy: 0.8258 - 2s/epoch - 13ms/step
    # 0.1 dr, 50 epochs, 64bs 94/94 - 1s - loss: 0.1078 - accuracy: 0.9643 - val_loss: 0.7735 - val_accuracy: 0.8017 - 1s/epoch - 14ms/step
    # 0.2 dr, 100 epochs,64 bs # 0.1 dr, 50 epochs, 64bs 94/94 - 1s - loss: 0.1078 - accuracy: 0.9643 - val_loss: 0.7735 - val_accuracy: 0.8017 - 1s/epoch - 14ms/step
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50, verbose=2)
    model.save("audio_classification/GTZAN_DB/models/GTZAN_LSTM_2.h5")
    # testing accuracy
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    #print(np.sum(y_pred == y_test) / len(y_pred))
    general.plot_hist(history)


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
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(units=Constants.NUM_CLASSES, activation='softmax')
    ])
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
    cnn_model.summary()
    # 188/188 - 2s - loss: 0.0239 - acc: 0.9680 - val_loss: 0.0694 - val_acc: 0.8992 - 2s/epoch - 9ms/step
    history = cnn_model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=40, verbose=2)
    cnn_model.save("audio_classification/GTZAN_DB/models/GTZAN_CNN_2.h5")
    # testing accuracy
    y_pred = cnn_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(np.sum(y_pred == y_test) / len(y_pred))

    general.plot_hist(history)


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
    #print(mfcc_data["duration"])
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
    values = []
    percentages = []
    for arr in y_pred:
        val, perc = general.three_most_frequent(arr)
        values.append(val)
        percentages.append(perc)
    return values, percentages








