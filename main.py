#
from Config import Constants
from spleeter.separator import Separator
from faster_whisper import WhisperModel
import warnings
warnings.filterwarnings('ignore')

def source_separation():
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(Constants.INPUT_AUDIO, Constants.OUTPUT_AUDIO)

def transcript():
    model_size = "large-v3"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe("audio.mp3", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    #model = whisper.load_model("base")
    #result = model.transcribe(Constants.INPUT_AUDIO)
    #with (open("Config/output_audio/transcription.txt", "w") as f):
    #    f.write(result["text"])

if __name__ == '__main__':
    source_separation()
    #transcript()




