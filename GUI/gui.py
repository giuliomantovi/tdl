import customtkinter
from customtkinter import filedialog
import tkinter
import pygame
from PIL import Image, ImageTk
from threading import *
import time
import math
import shutil
import os

from posixpath import join
from audio_processing import general
from Config import Constants

customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("400x480")
app.title("Music classifier")
pygame.mixer.init()
audios_list = []
n = 0


def select_file():
    global audios_list
    files_list = filedialog.askopenfilenames()
    # dirname = filedialog.askdirectory()
    new_dir_path = os.path.abspath(join(os.path.dirname(__file__), '..', Constants.INPUT_AUDIO))
    print(new_dir_path)
    for audio in files_list:
        new_audio_path = os.path.abspath(join(new_dir_path, audio.split("/")[-1]))
        print(new_audio_path)
        if not os.path.samefile(audio, new_audio_path):
            shutil.copy(audio, new_audio_path)

    general.convert_to_wav(new_dir_path)
    for root, subdirs, files in os.walk(new_dir_path):
        for filename in files:
            if filename.endswith(".wav"):
                print(join(root, filename))
                audios_list.append(os.path.join(root, filename))

    if audios_list:
        show_player()
    print(audios_list)


def progress():
    a = pygame.mixer.Sound(f'{audios_list[n]}')
    song_len = a.get_length() * 3
    for i in range(0, math.ceil(song_len)):
        time.sleep(.4)
        progressbar.set(pygame.mixer.music.get_pos() / 1000000)


def threading():
    t1 = Thread(target=progress)
    t1.start()


def play_music():
    threading()
    global n
    current_song = n
    if n > 2:
        n = 0
    song_name = audios_list[n]
    pygame.mixer.music.load(song_name)
    pygame.mixer.music.play(loops=0)
    pygame.mixer.music.set_volume(.5)
    # get_album_cover(song_name, n)

    # print('PLAY')
    n += 1


def skip_forward():
    # As an idea, you can turn play_music() into a start/pause function and create a seperate skip ahead function for this!
    play_music()


def skip_back():
    global n
    n -= 2
    play_music()


def volume(value):
    # print(value) # If you care to see the volume value in the terminal, un-comment this :)
    pygame.mixer.music.set_volume(value)


def show_player():
    play_button.place(relx=0.5, rely=0.7, anchor=customtkinter.CENTER)
    skip_f.place(relx=0.7, rely=0.7, anchor=customtkinter.CENTER)
    skip_b.place(relx=0.3, rely=0.7, anchor=customtkinter.CENTER)
    slider.place(relx=0.5, rely=0.78, anchor=customtkinter.CENTER)
    progressbar.place(relx=.5, rely=.85, anchor=customtkinter.CENTER)


# All Buttons
select_button = customtkinter.CTkButton(master=app, text="Select audios", command=select_file)
select_button.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

play_button = customtkinter.CTkButton(master=app, text='Play', command=play_music)

skip_f = customtkinter.CTkButton(master=app, text='>', command=skip_forward, width=2)

skip_b = customtkinter.CTkButton(master=app, text='<', command=skip_back, width=2)

slider = customtkinter.CTkSlider(master=app, from_=0, to=1, command=volume, width=210)

progressbar = customtkinter.CTkProgressBar(master=app, progress_color='#32a85a', width=250)

app.mainloop()
