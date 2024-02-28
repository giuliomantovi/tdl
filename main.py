import librosa
import numpy as np
import pandas as pd

import Recorder
from Config import Constants
from spleeter.separator import Separator
from faster_whisper import WhisperModel
import whisper
import pydub as pyd
import matplotlib.pyplot as plt
from model import *
from Recorder import Recorder


# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# import warnings

# warnings.filterwarnings('ignore')

def source_separation(file_path):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(file_path, Constants.OUTPUT_AUDIO)


def fast_transcript():
    model_size = "small"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(Constants.INPUT_AUDIO, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def simple_transcript(file_path):
    # WHISPER AI WITHOUT CUDA
    model = whisper.load_model("medium", download_root="whisper_models", in_memory=True)

    result = model.transcribe(file_path)
    with (open(Constants.OUTPUT_AUDIO + "transcription.txt", "w") as f):
        print(result["text"])
        f.write(result["text"])


def low_level_transcript():
    model = whisper.load_model("base", download_root="whisper_models", in_memory=True)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("Config/output_audio/lazza/vocals.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    print("model device: " + model.device.__str__())
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)


def create_spectrogram(audio):
    y, sr = librosa.load(audio)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    img = librosa.display.specshow(S_db, ax=ax)
    plt.show()


def create_mel_spectrogram(audio_path, filename):
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


# Create function to convert audio file to wav
def convert_to_wav(dir_path):
    """Takes a folder of audio file of non .wav format and converts to .wav"""
    for root, subdirs, files in os.walk(dir_path):
        for filename in files:
            if filename[-4:] == ".wav":
                continue
            if not os.path.isdir(filename):
                audio = pyd.AudioSegment.from_file(os.path.join(root, filename))
                new_filename = filename.split(".")[0] + ".wav"
                new_filename = os.path.join(root, new_filename)
                print(new_filename)
                audio.export(new_filename, format="wav")
                #print(f"Converting {filename} to {new_filename}...")


"""def extract_genre_subdirectories(file_path, num_subdirectories=5):
    subdirectories = []
    for _ in range(num_subdirectories):
        file_path, subdirectory = os.path.split(file_path)
        subdirectories.insert(0, subdirectory)
    return subdirectories[3]"""

if __name__ == '__main__':
    """mic = Recorder()
    mic.setMicrophone()
    mic.record()"""
    # source_separation(Constants.INPUT_AUDIO + "/country/ringoffire.wav")
    # simple_transcript(Constants.OUTPUT_AUDIO + "ringoffire/vocals.wav")

    # data=preprocess_dataset(Constants.INPUT_AUDIO)
    # create_pretrained_efficientnet_model(Constants.GTZAN_IMAGE_PATH)
    testefficientnetmodel(Constants.INPUT_IMAGES, Constants.EFFICIENTNET_PRETRAINED_PATH)
    # convert_to_wav(Constants.INPUT_AUDIO)
    # model_build_crnn6(Constants.GTZAN_IMAGE_PATH)
    # audio_to_spectrograms(Constants.INPUT_AUDIO)
