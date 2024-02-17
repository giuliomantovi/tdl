import Recorder
from Config import Constants
from spleeter.separator import Separator
from faster_whisper import WhisperModel
import whisper
from Recorder import Recorder
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# import warnings

# warnings.filterwarnings('ignore')

def source_separation():
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(Constants.INPUT_AUDIO + "mic_input.wav", Constants.OUTPUT_AUDIO)

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


def simple_transcript():
    # WHISPER AI WITHOUT CUDA
    model = whisper.load_model("medium", download_root="whisper_models", in_memory=True)

    result = model.transcribe(Constants.INPUT_AUDIO + "mic_input.wav")
    with (open(Constants.OUTPUT_AUDIO + "transcription.txt", "w") as f):
        print(result["text"])
        f.write(result["text"])

def low_level_transcript():
    model = whisper.load_model("base", download_root="whisper_models", in_memory=True)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("Config/output_audio/lazza/vocals.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    print("model device: "+model.device.__str__())
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

if __name__ == '__main__':
    """mic = Recorder()
    mic.setMicrophone()
    mic.record()"""
    #source_separation()
    #simple_transcript()