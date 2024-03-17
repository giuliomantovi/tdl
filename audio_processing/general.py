import os
import pydub as pyd


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
                # print(f"Converting {filename} to {new_filename}...")


"""def extract_genre_subdirectories(file_path, num_subdirectories=5):
    subdirectories = []
    for _ in range(num_subdirectories):
        file_path, subdirectory = os.path.split(file_path)
        subdirectories.insert(0, subdirectory)
    return subdirectories[3]"""
