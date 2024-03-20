import customtkinter
from customtkinter import filedialog
from tkinter import *
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
app.geometry("700x512")
app.title("Music classifier")
pygame.mixer.init()
audios_paths_list = []
audios_names_list = []
n = 0  # index of current song
stopped = 1  # flag for music player

forward = customtkinter.CTkImage(Image.open("icons/forward.png"))
backward = customtkinter.CTkImage(Image.open("icons/backward.png"))
play = customtkinter.CTkImage(Image.open("icons/play.png"))
stop = customtkinter.CTkImage(Image.open("icons/stop.png"))

def load_audios(new_dir_path):
    global audios_paths_list, audios_names_list
    for root, subdirs, files in os.walk(new_dir_path):
        print("see")
        for filename in files:
            new_filename = os.path.abspath(join(root, filename)).replace("\\", "/")
            if filename.endswith(".wav") and new_filename not in audios_paths_list:
                audios_paths_list.append(new_filename)
                audios_names_list.append(filename[:-4])
    if audios_names_list:
        songs_combobox.configure(values=audios_names_list)
        songs_combobox.set(audios_names_list[0])


def select_file():
    files_list = filedialog.askopenfilenames()
    root_dir = os.path.dirname(__file__)
    new_dir_path = os.path.abspath(join(root_dir, '..', Constants.INPUT_AUDIO))
    for audio in files_list:
        new_audio_path = os.path.normpath(join(new_dir_path, audio.split("/")[-1]))
        if not os.path.normpath(audio) == new_audio_path:
            shutil.copy(audio, new_audio_path)
    general.convert_to_wav(new_dir_path)
    load_audios(new_dir_path)
    # enabling components state
    """if audios_paths_list:
        show_player()
    print(audios_paths_list)"""


"""def progress():
    a = pygame.mixer.Sound(f'{audios_paths_list[n]}')
    song_len = a.get_length()
    print(math.ceil(song_len))
    for i in range(0, math.ceil(song_len*10/4)):
        time.sleep(0.4)
        print(pygame.mixer.music.get_pos()/(1000*song_len))
        progressbar.set(pygame.mixer.music.get_pos()/(1000*song_len))


def threading():
    t1 = Thread(target=progress, daemon=True)
    t1.start()"""


def update_progressbar():
    a = pygame.mixer.Sound(f'{audios_paths_list[n]}')
    song_len = a.get_length() * 1000
    bar_pos = pygame.mixer.music.get_pos()
    if bar_pos < song_len:
        progressbar.set(bar_pos / song_len)
    app.after(500, update_progressbar)

def play_music():
    #threading()
    update_progressbar()
    play_button.configure(image=stop)
    song_name = audios_paths_list[n]
    pygame.mixer.music.load(song_name)
    pygame.mixer.music.play(loops=0)
    pygame.mixer.music.set_volume(.5)


def stop_music():
    global stopped
    stopped = 1
    pygame.mixer.music.pause()


def handle_player():
    global stopped
    if stopped == 1 and pygame.mixer.music.get_pos() >= 0:
        play_button.configure(image=stop)
        print("MUSIC STOPPED")
        stopped = 0
        pygame.mixer.music.unpause()
    elif stopped == 1 and pygame.mixer.music.get_pos() < 0:
        print("FIRST START")
        stopped = 0
        play_music()
    else:
        play_button.configure(image=play)
        print("MUSIC ONGOING")
        stop_music()




def skip_forward():
    global n, stopped
    stopped = 0
    n += 1
    if n >= len(audios_paths_list):
        n = 0
    # As an idea, you can turn play_music() into a start/pause function and create a separate skip ahead function for this!
    play_music()
    songs_combobox.set(audios_names_list[n])


def skip_back():
    global n, stopped
    stopped = 0
    n -= 1
    if n < 0:
        n = len(audios_paths_list) - 1
    play_music()
    songs_combobox.set(audios_names_list[n])

def volume(value):
    # print(value) # If you care to see the volume value in the terminal, un-comment this :)
    pygame.mixer.music.set_volume(value)


def change_song(choice):
    global n, stopped
    n = audios_names_list.index(choice)
    stopped = 0
    play_music()


def show_player():
    """play_button.configure(state="normal")
    skip_b.configure(state="normal")
    skip_f.configure(state="normal")
    slider.configure(state="normal")"""


# All Buttons

select_button = customtkinter.CTkButton(master=app, text="Add audios", command=select_file)
select_button.place(relx=0.5, rely=0.6, anchor=customtkinter.CENTER)

songs_label = customtkinter.CTkLabel(master=app, text="SONGS")
songs_label.pack(pady=40)

songs_combobox = customtkinter.CTkComboBox(master=app, command=change_song, state="readonly")
songs_combobox.pack(pady=20)



play_button = customtkinter.CTkButton(master=app, text="", image=play, command=handle_player, width=50)
play_button.place(relx=0.5, rely=0.7, anchor=customtkinter.CENTER)

skip_f = customtkinter.CTkButton(master=app, text="", image=forward, command=skip_forward, width=2)
skip_f.place(relx=0.6, rely=0.7, anchor=customtkinter.CENTER)

skip_b = customtkinter.CTkButton(master=app, text="", image=backward, command=skip_back, width=2)
skip_b.place(relx=0.4, rely=0.7, anchor=customtkinter.CENTER)

slider = customtkinter.CTkSlider(master=app, from_=0, to=1, command=volume, width=150)
slider.place(relx=0.5, rely=0.78, anchor=customtkinter.CENTER)

progressbar = customtkinter.CTkProgressBar(master=app, progress_color='#32a85a', width=180)
progressbar.place(relx=.5, rely=.85, anchor=customtkinter.CENTER)

load_audios("C:/Users/Utente/UNI/tesina_LAUREA/Config/input/user/audio")

app.mainloop()
