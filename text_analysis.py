import pandas as pd
import pickle


def print_pickle(file):
    # obj = pd.read_pickle(file)
    # print(obj)

    objects = []
    """with (open(file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break"""
    with (open(file, "rb")) as openfile:
        objects.append(pickle.load(openfile))
    print(objects)
