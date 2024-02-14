#
from Config import Constants
from spleeter.separator import Separator
from faster_whisper import WhisperModel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

warnings.filterwarnings('ignore')


def source_separation():
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(Constants.INPUT_AUDIO, Constants.OUTPUT_AUDIO)


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
    # WHISPER AI WITHOUT CUDA
    # model = whisper.load_model("base")
    # result = model.transcribe(Constants.INPUT_AUDIO)
    # with (open("Config/output_audio/transcription.txt", "w") as f):
    #    f.write(result["text"])


def slow_transcript():
    # WHISPER AI WITHOUT CUDA
    model = whisper.load_model("base")
    result = model.transcribe(Constants.INPUT_AUDIO)
    with (open("Config/output_audio/transcription.txt", "w") as f):
        f.write(result["text"])


if __name__ == '__main__':
    # source_separation()
    slow_transcript()
