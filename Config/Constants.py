# constants.py

"""This module defines project-level constants."""

NUM_CLASSES = 10

#relative path to dir containing input audio

INPUT_AUDIO = "Config/input/audio"

INPUT_IMAGES = "Config/input/images"

#relative path to dir that stores separated audios

OUTPUT_AUDIO = "Config/output_audio/"

GTZAN_AUDIO_PATH = "GTZAN/Data/genres_original"

GTZAN_IMAGE_PATH = "GTZAN/Data/images_original"

LSMT_PATH = "GTZAN/models/GTZAN_LSTM.h5"

CNN_PATH = "GTZAN/models/GTZAN_CNN.h5"

CNN_IMAGE_PATH = "GTZAN/models/GTZAN_IMAGE_CNN.h5"

EFFICIENTNET_PRETRAINED_PATH = "GTZAN/models/GTZAN_EFFICIENTNETB0.h5"

EFFICIENTNET_SCRATCH_PATH = "GTZAN/models/GTZAN_SCRATCH_EFFICIENTNETB0.h5"