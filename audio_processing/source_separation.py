from spleeter.separator import Separator
from audio_classification.mfcc_models import *


def source_separation(dir_path):
    separator = Separator('spleeter:2stems')
    for root, subdirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".wav"):
                separator.separate_to_file(os.path.join(root, filename), Constants.INPUT_AUDIO)