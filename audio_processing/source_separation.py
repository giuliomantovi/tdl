from spleeter.separator import Separator
from audio_classification.mfcc_models import *


def source_separation(dir_path):
    # creates a folder containing vocals and accompaniment for each .wav file found in dir_path
    try:
        separator = Separator('spleeter:2stems')
        for root, subdirs, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith(".wav"):
                    separator.separate_to_file(audio_descriptor=os.path.join(root, filename),
                                               destination=Constants.INPUT_AUDIO)
    except:
        print("source separation error, try again")

