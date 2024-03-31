import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import librosa
from Config import Constants
import sys

genre_dict = {0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "HipHop", 5: "Jazz",
              6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"}


def create_mel_spectrogram(dirpath, filename, model):
    # spectrogram with musical components (that humans can hear)
    # if image has already been created, skip
    if model == "CNN" and os.path.exists(os.path.join(dirpath, "cnn_spec", filename[:-4]) + ".png"):
        return
    elif model == "EffNet" and os.path.exists(os.path.join(dirpath, "effnet_spec", filename[:-4]) + ".png"):
        return
    y, sr = librosa.load(str(os.path.join(dirpath, filename)))
    s = librosa.feature.melspectrogram(y=y, sr=44100, hop_length=308, win_length=2205,
                                       n_mels=128, n_fft=4096, fmax=18000, norm='slaney')
    s_db_mel = librosa.amplitude_to_db(s, ref=np.max)
    print(s_db_mel.shape)
    effnet_figsize = (2, 2)
    CNN_figsize = (4.32, 2.88)

    if model == "CNN":
        fig, ax = plt.subplots(figsize=CNN_figsize)
        img = librosa.display.specshow(s_db_mel, ax=ax)
        plt.savefig(fname=os.path.join(dirpath, "cnn_spec", filename[:-4]) + ".png", format='png')

    else:
        fig, ax = plt.subplots(figsize=effnet_figsize)
        img = librosa.display.specshow(s_db_mel, ax=ax)
        plt.savefig(fname=os.path.join(dirpath, "effnet_spec", filename[:-4]) + ".png", format='png')

    # plt.show()


def audio_to_spectrograms(dir_path, model):
    for root, subdirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".wav"):
                # file_path = os.path.join(root, filename)
                create_mel_spectrogram(root, filename, model)


def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    index = np.argmax(counts)
    return unique[index]


def three_most_frequent(arr):
    #find 3 most frequent genres for mfcc based models
    values, counts = np.unique(arr, return_counts=True)
    total = sum(counts)
    different_val = min(3, len(values))
    ind = np.argpartition(-counts, kth=different_val-1)[:different_val]
    values = values[ind]
    occurrences = counts[ind]
    percentages = [x/total for x in occurrences]
    values = [genre_dict[x] for x in values]
    return values, percentages


def translate_predictions(predictions, names):
    #decodes prediction array for efficientnet model predictions
    genres = []
    percentages = []
    for i in range(len(predictions)):
        gen = []
        perc = []
        print(names[i].split(".")[0])
        three_largest, indexes = find3largest(predictions[i])
        tot = sum(three_largest)
        for j in range(len(three_largest)):
            three_largest[j] = round(three_largest[j] / tot, 2)
        for j in range(len(three_largest)):
            if three_largest[j] != 0.:
                gen.append(genre_dict[indexes[j]])
                perc.append(three_largest[j])
                print(genre_dict[indexes[j]] + ": " + str(int(three_largest[j])) + "%")
        genres.append(gen)
        percentages.append(perc)
    return genres, percentages


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

def set_plot():
    color = 'white'
    plt.rcParams['axes.edgecolor'] = color
    plt.rcParams['text.color'] = color
    plt.rcParams['legend.framealpha'] = 0.1
    plt.rcParams['axes.labelcolor'] = color
    plt.rcParams['xtick.color'] = color
    plt.rcParams['ytick.color'] = color
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 18


def plot_hist(hist):
    set_plot()
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    # plt.plot(hist.history["loss"])
    # plt.plot(hist.history["val_loss"])
    # plt.ylim(0.0, 1.0)
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    plt.clf()


def plot_logger(path):
    if not os.path.exists(path):
        return
    history = pd.read_csv(path, sep=',', engine='python')
    set_plot()
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    plt.plot(acc)
    plt.plot(val_acc)
    for var in (acc, val_acc):
        perc = round(var.max()*100, 1)
        plt.annotate(str(perc) + "%", xy=(1, var.max()), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.title("LSTM model")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(fname="C:/Users/Utente/UNI/tesina_LAUREA/GUI/images/ac_models/CNN_dark.png", format='png',
                bbox_inches="tight",
                 transparent=True)
    plt.clf()

"""def create_spectrogram(audio):
    y, sr = librosa.load(audio)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    img = librosa.display.specshow(S_db, ax=ax)
    plt.show()"""

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
