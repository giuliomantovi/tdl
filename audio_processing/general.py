import os
import pydub as pyd
from posixpath import join


def convert_to_wav(dir_path):
    """Takes a folder of audio file of non .wav format and converts to .wav"""
    for root, subdirs, files in os.walk(dir_path):
        for filename in files:
            if filename[-4:] == ".wav":
                continue
            old_filename = join(root, filename)
            if not os.path.isdir(filename):
                try:
                    audio = pyd.AudioSegment.from_file(old_filename)
                    new_filename = join(root, filename.split(".")[0] + ".wav")
                    #print(new_filename)
                    audio.export(new_filename, format="wav")
                    os.remove(old_filename)
                    # print(f"Converting {filename} to {new_filename}...")
                except:
                    print("Conversion error for file " + str(filename))
                    os.remove(old_filename)


"""def extract_genre_subdirectories(file_path, num_subdirectories=5):
    subdirectories = []
    for _ in range(num_subdirectories):
        file_path, subdirectory = os.path.split(file_path)
        subdirectories.insert(0, subdirectory)
    return subdirectories[3]"""
