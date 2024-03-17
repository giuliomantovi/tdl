# constants.py

"""This module defines project-level constants."""

NUM_CLASSES = 10

# relative path to dir containing input

INPUT_AUDIO = "Config/input/audio"

INPUT_IMAGES = "Config/input/images"

INPUT_TEXT = "Config/input/text"

# relative path to dir that stores separated audios

OUTPUT_AUDIO = "Config/output_audio/"

# dataset paths

GTZAN_AUDIO_PATH = "GTZAN_DB/Data/genres_original"

GTZAN_IMAGE_PATH = "GTZAN_DB/Data/images_original"

# model paths

LSMT_PATH = "GTZAN_DB/models/GTZAN_LSTM.h5"

CNN_PATH = "GTZAN_DB/models/GTZAN_CNN.h5"

CNN_IMAGE_PATH = "GTZAN_DB/models/GTZAN_IMAGE_CNN.h5"

EFFICIENTNET_PRETRAINED_PATH = "GTZAN_DB/models/GTZAN_EFFICIENTNETB0.h5"

EFFICIENTNET_SCRATCH_PATH = "GTZAN_DB/models/GTZAN_SCRATCH_EFFICIENTNETB0.h5"
