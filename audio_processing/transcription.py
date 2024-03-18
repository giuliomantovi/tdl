from audio_classification.mfcc_models import *
from faster_whisper import WhisperModel
import whisper


def fast_transcript(dir_path):
    # given a directory, creates transcriptions (vocals.txt) for all vocals.wav files in dir_path.
    try:
        model_size = "medium"
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16",
                             download_root=Constants.WHISPER_MODELS)
        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with FP16
        # model = WhisperModel(model_size, device="cpu", compute_type="float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")
        for root, subdirs, files in os.walk(dir_path):
            for filename in files:
                if filename == "vocals.wav":
                    segments, info = model.transcribe(str(os.path.join(root, filename)), beam_size=5)
                    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
                    with (open(os.path.join(root, filename[:-4]) + ".txt", "w") as f):
                        for segment in segments:
                            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                            # f.write("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                            f.write(" %s" % segment.text)
    except:
        print("Transcription error, try again")


def simple_transcript(file_path):
    # WHISPER AI WITHOUT CUDA
    model = whisper.load_model("medium", download_root="audio_prcessing/whisper_models", in_memory=True)

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
