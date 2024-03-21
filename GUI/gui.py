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

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

# self = customtkinter.CTk()  # create CTk window like you do with the Tk window


class App(customtkinter.CTk):
    pygame.mixer.init()
    audios_paths_list = []
    audios_names_list = []
    n = 0  # index of current song
    stopped = 1  # flag for music player

    forward = customtkinter.CTkImage(Image.open("icons/forward.png"))
    backward = customtkinter.CTkImage(Image.open("icons/backward.png"))
    play = customtkinter.CTkImage(Image.open("icons/play.png"))
    stop = customtkinter.CTkImage(Image.open("icons/stop.png"))

    #songs_combobox = 0

    def __init__(self):
        super().__init__()
        self.geometry("700x512")
        self.title("Music classifier")
        # configure grid layout (2x2)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1), weight=1)

        # create frame for music player
        self.sidebar_frame = customtkinter.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # All Buttons
        self.select_button = customtkinter.CTkButton(master=self.sidebar_frame, text="Add audios", command=self.select_file)
        # select_button.place(relx=0.5, rely=0.2, anchor=customtkinter.CENTER)
        self.select_button.grid(row=0, column=0, padx=20, pady=10)
        """songs_label = customtkinter.CTkLabel(master=self, text="SONGS")
        songs_label.pack()"""

        self.songs_combobox = customtkinter.CTkComboBox(master=self.sidebar_frame, width=150, command=self.change_song,
                                                        state="readonly")
        # songs_combobox.pack(pady=20, anchor=customtkinter.CENTER)
        self.songs_combobox.grid(row=1, column=0, padx=20, pady=10)

        self.play_button = customtkinter.CTkButton(master=self.sidebar_frame, text="", image=self.play, command=self.handle_player,
                                                   width=50)
        # play_button.place(relx=0.5, rely=0.3, anchor=customtkinter.CENTER)
        self.play_button.grid(row=2, column=0, padx=20, pady=10)

        self.skip_f = customtkinter.CTkButton(master=self.sidebar_frame, text="", image=self.forward, command=self.skip_forward, width=2)
        # skip_f.place(relx=0.6, rely=0.3, anchor=customtkinter.CENTER)
        self.skip_f.grid(row=2, column=0, padx=20, pady=10)

        self.skip_b = customtkinter.CTkButton(master=self.sidebar_frame, text="", image=self.backward, command=self.skip_back, width=2)
        # skip_b.place(relx=0.4, rely=0.3, anchor=customtkinter.CENTER)
        self.skip_b.grid(row=2, column=0, padx=20, pady=10)

        self.slider = customtkinter.CTkSlider(master=self.sidebar_frame, from_=0, to=1, command=self.volume, width=150)
        # slider.place(relx=0.5, rely=0.38, anchor=customtkinter.CENTER)
        self.slider.grid(row=3, column=0, padx=20, pady=10)

        self.progressbar = customtkinter.CTkProgressBar(master=self.sidebar_frame, progress_color='#32a85a', width=180)
        # progressbar.place(relx=.5, rely=.45, anchor=customtkinter.CENTER)
        self.progressbar.grid(row=4, column=0, padx=20, pady=10)

        self.load_audios("C:/Users/Utente/UNI/tesina_LAUREA/Config/input/user/audio")
        self.update_progressbar()

    def load_audios(self, new_dir_path):
        for root, subdirs, files in os.walk(new_dir_path):
            print("see")
            for filename in files:
                new_filename = os.path.abspath(join(root, filename)).replace("\\", "/")
                if filename.endswith(".wav") and new_filename not in self.audios_paths_list:
                    self.audios_paths_list.append(new_filename)
                    self.audios_names_list.append(filename[:-4])
        if self.audios_names_list:
            self.songs_combobox.configure(values=self.audios_names_list)
            self.songs_combobox.set(self.audios_names_list[0])

    def select_file(self):
        files_list = filedialog.askopenfilenames()
        root_dir = os.path.dirname(__file__)
        new_dir_path = os.path.abspath(join(root_dir, '..', Constants.INPUT_AUDIO))
        for audio in files_list:
            new_audio_path = os.path.normpath(join(new_dir_path, audio.split("/")[-1]))
            if not os.path.normpath(audio) == new_audio_path:
                shutil.copy(audio, new_audio_path)
        general.convert_to_wav(new_dir_path)
        self.load_audios(new_dir_path)
        # enabling components state
        """if audios_paths_list:
            show_player()
        print(audios_paths_list)"""

    def update_progressbar(self):
        a = pygame.mixer.Sound(f'{self.audios_paths_list[self.n]}')
        song_len = a.get_length() * 1000
        bar_pos = pygame.mixer.music.get_pos()
        if bar_pos < song_len:
            self.progressbar.set(bar_pos / song_len)
        self.after(500, self.update_progressbar)

    def play_music(self):
        self.play_button.configure(image=self.stop)
        song_name = self.audios_paths_list[self.n]
        pygame.mixer.music.load(song_name)
        pygame.mixer.music.play(loops=0)
        pygame.mixer.music.set_volume(.5)

    def stop_music(self):
        self.stopped = 1
        pygame.mixer.music.pause()

    def handle_player(self):
        if self.stopped == 1 and pygame.mixer.music.get_pos() >= 0:
            self.play_button.configure(image=self.stop)
            print("MUSIC STOPPED")
            self.stopped = 0
            pygame.mixer.music.unpause()
        elif self.stopped == 1 and pygame.mixer.music.get_pos() < 0:
            print("FIRST START")
            self.stopped = 0
            self.play_music()
        else:
            self.play_button.configure(image=self.play)
            print("MUSIC ONGOING")
            self.stop_music()

    def skip_forward(self):
        self.stopped = 0
        self.n += 1
        if self.n >= len(self.audios_paths_list):
            self.n = 0
        self.play_music()
        self.songs_combobox.set(self.audios_names_list[self.n])

    def skip_back(self):
        self.stopped = 0
        self.n -= 1
        if self.n < 0:
            self.n = len(self.audios_paths_list) - 1
        self.play_music()
        self.songs_combobox.set(self.audios_names_list[self.n])

    def volume(self, value):
        # print(value) # If you care to see the volume value in the terminal, un-comment this :)
        pygame.mixer.music.set_volume(value)

    def change_song(self, choice):
        self.n = self.audios_names_list.index(choice)
        self.stopped = 0
        self.play_music()

    def show_player(self):
        """play_button.configure(state="normal")
        skip_b.configure(state="normal")
        skip_f.configure(state="normal")
        slider.configure(state="normal")"""


if __name__ == "__main__":
    self = App()
    self.mainloop()
