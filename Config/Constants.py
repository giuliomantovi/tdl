# constants.py

"""This module defines project-level constants."""

NUM_CLASSES = 10

# relative path to dir containing input

INPUT_AUDIO = "Config/input/user/audio"

INPUT_IMAGES_CNN = "Config/input/user/images/cnn"
INPUT_IMAGES_EFFNET = "Config/input/user/images/effnet"

INPUT_TEXT = "Config/input/text"

# relative path to dir that stores separated audios

OUTPUT_AUDIO = "Config/output_audio/"

# dataset paths

GTZAN_AUDIO_PATH = "audio_classification/GTZAN_DB/Data/genres_original"

GTZAN_IMAGE_PATH = "audio_classification/GTZAN_DB/Data/images_original"

GENIUS_DATASET_PATH = "lyrics_classification/Genius_song_lyrics_DB/song_lyrics.csv"

# model paths

WHISPER_MODELS = "audio_processing/whisper_models"

SCRATCH_LDA_MODEL = "lyrics_classification/Genius_song_lyrics_DB/lda_model/lda_mod"
SCRATCH_LDA_DICTIONARY = "lyrics_classification/Genius_song_lyrics_DB/lda_model/lda_mod.id2word"

WASABI_LDA_MODEL = "lyrics_classification/WASABI_DB/topics/lda_model_16.jl"
WASABI_LDA_DICTIONARY = "lyrics_classification/WASABI_DB/topics/dictionary.pickle"

LSMT_PATH = "audio_classification/GTZAN_DB/models/GTZAN_LSTM.h5"

CNN_PATH = "audio_classification/GTZAN_DB/models/GTZAN_CNN.h5"

CNN_IMAGE_PATH = "audio_classification/GTZAN_DB/models/GTZAN_IMAGE_CNN.h5"

EFFICIENTNET_PRETRAINED_PATH = "audio_classification/GTZAN_DB/models/GTZAN_EFFICIENTNETB0.h5"

EFFICIENTNET_SCRATCH_PATH = "audio_classification/GTZAN_DB/models/GTZAN_SCRATCH_EFFICIENTNETB0.h5"
