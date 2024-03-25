import matplotlib.pyplot as plt
import numpy as np


def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    index = np.argmax(counts)
    return unique[index]


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    # plt.plot(hist.history["loss"])
    # plt.plot(hist.history["val_loss"])
    # plt.ylim(0.0, 1.0)
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


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
