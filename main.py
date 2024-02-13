#
from Config import Constants
from spleeter.separator import Separator
import warnings
warnings.filterwarnings('ignore')
def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')
    separator = Separator('spleeter:5stems')
    separator.separate_to_file(Constants.INPUT_AUDIO, Constants.OUTPUT_AUDIO)


