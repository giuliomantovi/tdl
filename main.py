#
from Config import Constants
from spleeter.separator import Separator
import whisper
import warnings
warnings.filterwarnings('ignore')

def source_separation():
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(Constants.INPUT_AUDIO, Constants.OUTPUT_AUDIO)

def transcript():
    model = whisper.load_model("base")
    result = model.transcribe(Constants.INPUT_AUDIO)
    with open("Config/output_audio/transcription.txt", "w") as f: f.write(result["text"])

if __name__ == '__main__':
    #source_separation()
    transcript()



