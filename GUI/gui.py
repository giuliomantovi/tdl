import customtkinter
from customtkinter import filedialog
# from tkinter import *
import tkinter
import pygame
from PIL import Image, ImageTk
from threading import *
import time
import math
import shutil
import os
from posixpath import join
import matplotlib.pyplot as plt
import numpy as np

from audio_processing import Recorder

from Config import Constants

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


# self = customtkinter.CTk()  # create CTk window like you do with the Tk window


class App(customtkinter.CTk):
    pygame.mixer.init()
    audios_paths_list = []
    audios_names_list = []
    n = 0  # index of current song
    stopped = 1  # flag for music player
    recorder = Recorder.Recorder()
    recording = 0  # flag for tracking recording

    forward = customtkinter.CTkImage(Image.open("icons/forward.png"))
    backward = customtkinter.CTkImage(Image.open("icons/backward.png"))
    play = customtkinter.CTkImage(Image.open("icons/play.png"))
    stop = customtkinter.CTkImage(Image.open("icons/stop.png"))
    microphone = customtkinter.CTkImage(Image.open("icons/microphone.png"))
    audios_dir = os.path.abspath(join(os.getcwd(), '..', Constants.INPUT_AUDIO))

    # songs_combobox = 0

    def __init__(self):

        super().__init__()
        self.geometry(f"{1050}x{600}")
        self.title("Music classifier")
        # configure grid layout (2x2)
        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1), weight=1)

        # region left sidebar
        # LEFT SIDEBAR
        self.sidebar_frame = customtkinter.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        # MUSIC PLAYER LABEL
        self.songs_label = customtkinter.CTkLabel(master=self.sidebar_frame, text="MUSIC PLAYER",
                                                  font=("Helvetica", 18))
        self.songs_label.grid(row=0, column=0, padx=5, pady=(20, 0))
        # SONG CHOICE COMBOBOX
        self.songs_combobox = customtkinter.CTkComboBox(master=self.sidebar_frame, width=150, command=self.change_song,
                                                        state="readonly", font=("Helvetica", 16))
        self.songs_combobox.grid(row=1, column=0, padx=20, pady=10)
        # HORIZONTAL GRID FOR 3 BUTTONS
        self.buttons_grid = customtkinter.CTkFrame(master=self.sidebar_frame, width=50, corner_radius=0,
                                                   fg_color='transparent')
        self.buttons_grid.grid(row=2, column=0, sticky="nsew")
        # PREVIOUS SONG BUTTON
        self.skip_b = customtkinter.CTkButton(master=self.buttons_grid, text="", image=self.backward,
                                              command=self.skip_back, width=50)
        self.skip_b.grid(row=0, column=0, padx=(35, 5), pady=10)
        # PLAY SONG BUTTON
        self.play_button = customtkinter.CTkButton(master=self.buttons_grid, text="", image=self.play,
                                                   command=self.handle_player, width=50)
        self.play_button.grid(row=0, column=1, padx=5, pady=10)
        # NEXT SONG BUTTON
        self.skip_f = customtkinter.CTkButton(master=self.buttons_grid, text="", image=self.forward,
                                              command=self.skip_forward, width=50)
        self.skip_f.grid(row=0, column=2, padx=5, pady=10)
        # VOLUME SLIDER
        self.slider = customtkinter.CTkSlider(master=self.sidebar_frame, from_=0, to=1, command=self.volume, width=150)
        self.slider.grid(row=3, column=0, padx=20, pady=(5, 15))
        # SONG TIME PROGRESS BAR
        self.progressbar = customtkinter.CTkProgressBar(master=self.sidebar_frame, progress_color='#32a85a', width=180)
        self.progressbar.grid(row=4, column=0, padx=20, pady=5)
        # HORIZONTAL GRID FOR NEXT 2 BUTTONS
        self.selectfile_grid = customtkinter.CTkFrame(master=self.sidebar_frame, corner_radius=0,
                                                      fg_color='transparent')
        self.selectfile_grid.grid(row=5, column=0, sticky="nsew")
        # ADD AUDIOS SELECTION BUTTONS
        self.select_button = customtkinter.CTkButton(master=self.selectfile_grid, text="Add audios",
                                                     font=("Helvetica", 16),
                                                     command=self.select_file, width=130)
        self.select_button.grid(row=0, column=0, padx=(30, 5), pady=20)
        self.register_button = customtkinter.CTkButton(master=self.selectfile_grid, text="", image=self.microphone,
                                                       command=self.handle_recording, width=30)
        self.register_button.grid(row=0, column=1, padx=5, pady=20)
        # SOURCE SEPARATION LABEL
        """self.songs_label = customtkinter.CTkLabel(master=self.sidebar_frame, text="SOURCE SEPARATION")
        self.songs_label.grid(row=7, column=0, padx=5, pady=(20, 0))

        # PLAY VOCALS AND ACCOMPANIMENT BUTTONS
        self.vocals_button = customtkinter.CTkButton(master=self.sidebar_frame, text="Play Vocals",
                                                     command=self.play_vocals#, state="disabled"
        )
        self.vocals_button.grid(row=8, column=0, padx=20, pady=5)
        self.accompaniment_button = customtkinter.CTkButton(master=self.sidebar_frame, text="Play Accompaniment",
                                                            command=self.play_accompaniment#, state="disabled"
        )
        self.accompaniment_button.grid(row=9, column=0, padx=20, pady=5)"""
        # TRANSCRIPTION LABEL
        self.songs_label = customtkinter.CTkLabel(master=self.sidebar_frame, text="TRANSCRIPTION",
                                                  font=("Helvetica", 18))
        self.songs_label.grid(row=6, column=0, padx=5, pady=(10, 0))
        # COMPUTE TRANSCRIPTIONS BUTTON
        self.select_button = customtkinter.CTkButton(master=self.sidebar_frame, text="Create transcriptions",
                                                     font=("Helvetica", 16),
                                                     command=self.transcription_threading)
        self.select_button.grid(row=7, column=0, padx=20, pady=5)
        # TRANSCRIPTION TEXTBOX
        self.lyrics_textbox = customtkinter.CTkTextbox(master=self.sidebar_frame, wrap='word', font=("Helvetica", 16))
        self.lyrics_textbox.grid(row=8, column=0, padx=20, pady=(15, 10))
        # LOADING OPERATIONS
        self.load_audios(os.path.abspath(join(os.getcwd(), '..', Constants.INPUT_AUDIO)))
        self.update_progressbar()
        # endregion left sidebar

        # CREATE TABLEVIEW FOR 2ND PART OF THE APP
        self.tabview = customtkinter.CTkTabview(self, fg_color="transparent")
        self.tabview._segmented_button.configure(font=("Helvetica", 16))
        self.tabview.grid(row=0, column=1, rowspan=2, padx=(15, 15), pady=(5, 5), sticky="nsew")
        self.tabview.add("Audio classification")
        self.tabview.add("Lyrics classification")
        self.tabview.add("Songs similarity")
        # self.tabview.tab("Audio classification").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        # self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)

        """
        -----------------------------------------------------------------------------------------------------------
                                        AUDIO CLASSIFICATION TAB
        -----------------------------------------------------------------------------------------------------------
        """
        # region audio classification tab
        # create radiobutton frame for ac
        self.ac_radiobutton_frame = customtkinter.CTkFrame(self.tabview.tab("Audio classification"))
        self.ac_radiobutton_frame.grid(row=0, column=0, padx=(20, 20), pady=(20, 5), sticky="nsew")
        """# HORIZONTAL GRID FOR RADIO BUTTONS(MODEL CHOICE) AND MODEL DESCRIPTIONS
        self.ac_models_grid = customtkinter.CTkFrame(master=self.sidebar_frame, corner_radius=0,
                                                     fg_color='transparent')
        self.ac_models_grid.grid(row=0, column=0, sticky="nsew")"""
        # inserting radiobuttons
        self.ac_model_selected = customtkinter.StringVar(value="LSTM")
        self.ac_label_radio_group = customtkinter.CTkLabel(master=self.ac_radiobutton_frame, text="MODEL SELECTION",
                                                           font=("Helvetica", 18))
        self.ac_label_radio_group.grid(row=0, column=0, columnspan=1, padx=10, pady=20, sticky="")
        self.ac_radio_button_lsmt = customtkinter.CTkRadioButton(master=self.ac_radiobutton_frame, text="LSTM",
                                                                 variable=self.ac_model_selected, value="LSTM",
                                                                 command=self.pressed_ac_radiobutton,
                                                                 font=("Helvetica", 16))
        self.ac_radio_button_lsmt.grid(row=1, column=0, pady=(5, 10), padx=20, sticky="w")
        self.ac_radio_button_cnn = customtkinter.CTkRadioButton(master=self.ac_radiobutton_frame, text="CNN - mfcc",
                                                                variable=self.ac_model_selected, value="CNN",
                                                                command=self.pressed_ac_radiobutton,
                                                                font=("Helvetica", 16))
        self.ac_radio_button_cnn.grid(row=2, column=0, pady=15, padx=20, sticky="w")
        self.ac_radio_button_cnnspec = customtkinter.CTkRadioButton(master=self.ac_radiobutton_frame,
                                                                    text="CNN - spectrogram",
                                                                    variable=self.ac_model_selected,
                                                                    value="CNN spectrogram",
                                                                    command=self.pressed_ac_radiobutton,
                                                                    font=("Helvetica", 16))
        self.ac_radio_button_cnnspec.grid(row=3, column=0, pady=15, padx=20, sticky="w")
        self.ac_radio_button_effnet = customtkinter.CTkRadioButton(master=self.ac_radiobutton_frame,
                                                                   text="EfficientNet (suggested)",
                                                                   variable=self.ac_model_selected,
                                                                   value="EfficientNet",
                                                                   command=self.pressed_ac_radiobutton,
                                                                   font=("Helvetica", 16))
        self.ac_radio_button_effnet.grid(row=4, column=0, pady=(15, 0), padx=20, sticky="w")
        # Images for models description
        # 662 x 480
        self.ac_images_size = (450, 270)
        self.ac_lsmt_image = customtkinter.CTkImage(dark_image=Image.open(
            "images/ac_models/LSTM_dark.png"), size=self.ac_images_size)
        self.ac_cnn_image = customtkinter.CTkImage(dark_image=Image.open(
            "images/ac_models/CNN_dark.png"), size=self.ac_images_size)
        self.ac_cnnImage_image = customtkinter.CTkImage(dark_image=Image.open(
            "images/ac_models/CNN_IMAGE_dark.png"), size=self.ac_images_size)
        self.ac_effNet_image = customtkinter.CTkImage(dark_image=Image.open(
            "images/ac_models/EFFNET_dark.png"), size=self.ac_images_size)
        # displaying initial model image
        self.ac_model_image = customtkinter.CTkLabel(master=self.tabview.tab("Audio classification"),
                                                     image=self.ac_lsmt_image, text="")
        self.ac_model_image.grid(row=0, column=1, padx=(25, 5), pady=(20, 5), sticky="n")
        # SONG CHOICE LABEL AND COMBOBOX 2
        self.ac_songs_label = customtkinter.CTkLabel(master=self.tabview.tab("Audio classification"),
                                                     text="SONG GENRE PREDICTION", font=("Helvetica", 18))
        self.ac_songs_label.grid(row=1, column=0, padx=(5, 5), pady=(20, 5))
        self.ac_songs_combobox = customtkinter.CTkComboBox(master=self.tabview.tab("Audio classification"),
                                                           state="readonly", values=self.audios_names_list,
                                                           font=("Helvetica", 16), command=self.ac_change_song)
        self.ac_songs_combobox.grid(row=2, column=0, padx=10, pady=(35, 35))
        if self.audios_paths_list:
            self.ac_songs_combobox.set(self.audios_names_list[0])
        # GENRE PREDICTION BUTTON AND IMAGE
        self.ac_predict_button = customtkinter.CTkButton(master=self.tabview.tab("Audio classification"),
                                                         text="Create genre predictions",
                                                         command=self.ac_predict_genre,
                                                         font=("Helvetica", 16))
        self.ac_predict_button.grid(row=3, column=0, padx=10, pady=(15, 5))
        self.ac_prediction_image = customtkinter.CTkLabel(master=self.tabview.tab("Audio classification"),
                                                          image=None, text="")
        self.ac_prediction_image.grid(row=1, column=1, rowspan=3, padx=(0, 0), pady=(20, 0), sticky="n")
        ac_selected_song = self.ac_songs_combobox.get()
        ac_selected_image = join(self.audios_dir, "genre_predictions", ac_selected_song) + ".png"
        if os.path.exists(ac_selected_image):
            ac_genres_image = customtkinter.CTkImage(dark_image=Image.open(ac_selected_image), size=(256, 192))
            self.ac_prediction_image.configure(image=ac_genres_image)
        # endregion audio classification tab

        """
        -----------------------------------------------------------------------------------------------------------
                                        LYRICS CLASSIFICATION TAB
        -----------------------------------------------------------------------------------------------------------
        """
        # region lyrics classification tab
        #self.tabview.tab("Lyrics classification").grid_columnconfigure(1, weight=1)

        self.lc_radiobutton_frame = customtkinter.CTkFrame(self.tabview.tab("Lyrics classification"))
        self.lc_radiobutton_frame.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        # inserting radiobuttons
        self.lc_model_selected = customtkinter.StringVar(value="Scratch LDA")
        self.lc_label_radio_group = customtkinter.CTkLabel(master=self.lc_radiobutton_frame, text="MODEL SELECTION",
                                                           font=("Helvetica", 18))
        self.lc_label_radio_group.grid(row=0, column=0, columnspan=1, padx=10, pady=20, sticky="")
        self.lc_radio_button_pretr_lda = customtkinter.CTkRadioButton(master=self.lc_radiobutton_frame,
                                                                      text="Scratch LDA",
                                                                      variable=self.lc_model_selected,
                                                                      value="Scratch LDA",
                                                                      command=self.pressed_lc_radiobutton,
                                                                      font=("Helvetica", 16))
        self.lc_radio_button_pretr_lda.grid(row=1, column=0, pady=(5, 10), padx=20, sticky="w")
        self.lc_radio_button_scratch_lda = customtkinter.CTkRadioButton(master=self.lc_radiobutton_frame,
                                                                        text="Pretrained LDA",
                                                                        variable=self.lc_model_selected,
                                                                        value="Pretrained LDA",
                                                                        command=self.pressed_lc_radiobutton,
                                                                        font=("Helvetica", 16))
        self.lc_radio_button_scratch_lda.grid(row=2, column=0, pady=15, padx=20, sticky="w")

        # Scrollable frame for model topics
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self.tabview.tab("Lyrics classification"),
                                                                 label_text="Topics")
        self.scrollable_frame.grid(row=0, column=1, padx=(10, 10), pady=(20, 20), sticky="w")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame_buttons = []
        for i in range(4):
            button = customtkinter.CTkButton(master=self.scrollable_frame, text=f"Topic {i}")
            button.grid(row=i, column=0, padx=10, pady=(0, 20))
            self.scrollable_frame_buttons.append(button)

        # topic image
        # Images for models description
        # 662 x 480
        self.ac_images_size = (250, 150)
        self.lc_image = customtkinter.CTkImage(dark_image=Image.open(
            "images/lc_models/scratch_lda/cloud_topic0.png"), size=self.ac_images_size)
        # displaying initial model image
        self.lc_topic_image = customtkinter.CTkLabel(master=self.tabview.tab("Lyrics classification"),
                                                     image=self.lc_image, text="")
        self.lc_topic_image.grid(row=0, column=2, padx=(15, 5), pady=(5, 5), sticky="w")
        # endregion lyrics classification tab

    """
    -----------------------------------------------------------------------------------------------------------
                                        AUDIO CLASSIFICATION FUNCTIONS
    -----------------------------------------------------------------------------------------------------------
    """

    # region audio classification functions
    def pressed_ac_radiobutton(self):
        match self.ac_model_selected.get():
            case "EfficientNet":
                self.ac_model_image.configure(image=self.ac_effNet_image)
            case "CNN":
                self.ac_model_image.configure(image=self.ac_cnn_image)
            case "CNN spectrogram":
                self.ac_model_image.configure(image=self.ac_cnnImage_image)
            case "LSTM":
                self.ac_model_image.configure(image=self.ac_lsmt_image)

    def ac_change_song(self, choice):
        selected_image = join(self.audios_dir, "genre_predictions", choice) + ".png"
        ac_genres_image = customtkinter.CTkImage(dark_image=Image.open(selected_image), size=(256, 192))
        self.ac_prediction_image.configure(image=ac_genres_image)

    def ac_predict_genre(self):
        from audio_classification import general
        self.ac_prediction_image.configure(image='')
        self.ac_prediction_image.configure(text="Loading...")
        self.ac_prediction_image.update()

        model = self.ac_model_selected.get()
        genres = percentages = []
        match model:
            case "EfficientNet":
                from audio_classification import efficientnet_model
                print("effnet")
                general.audio_to_spectrograms(self.audios_dir, "EffNet")
                model_path = os.path.abspath(join(os.getcwd(), '..', Constants.EFFICIENTNET_PRETRAINED_PATH))
                images_path = join(self.audios_dir, "effnet_spec")
                genres, percentages = efficientnet_model.testefficientnetmodel(images_path, model_path)
            case "CNN spectrogram":
                from audio_classification import spectrogram_models
                print("CNN IMAGE")
                general.audio_to_spectrograms(self.audios_dir, "cnn")
                model_path = os.path.abspath(join(os.getcwd(), '..', Constants.CNN_IMAGE_PATH))
                images_path = join(self.audios_dir, "cnn_spec")
                genres, percentages = spectrogram_models.testimagemodel(images_path, model_path)
            case "CNN" | "LSTM":
                from audio_classification import mfcc_models
                print("CNN/LSMT")
                data = mfcc_models.preprocess_dir(self.audios_dir)
                if model == "CNN":
                    model_path = Constants.CNN_PATH
                else:
                    model_path = Constants.LSMT_PATH
                model_path = os.path.abspath(join(os.getcwd(), '..', model_path))
                genres, percentages = mfcc_models.testaudiomodel(data, model_path)
            case _:
                print("Unknown error, change model")
        print(" GENRES: ")
        print(genres)
        print("PERC: ")
        print(percentages)
        print(model)
        self.ac_create_predictions_plot(model, genres, percentages)
        self.ac_prediction_image.configure(text="")
        self.ac_change_song(self.ac_songs_combobox.get())
        self.ac_predict_button.configure(state="normal")
        # plt.show()

    def ac_create_predictions_plot(self, model, genres, percentages):
        color = 'white'
        plt.rcParams['text.color'] = color
        plt.rcParams['axes.edgecolor'] = color
        plt.rcParams['axes.labelcolor'] = color
        plt.rcParams['xtick.color'] = color
        plt.rcParams['ytick.color'] = color
        plt.rcParams['font.size'] = 22

        for i in range(len(genres)):
            plt.clf()
            x = np.array(genres[i])
            y = np.array(percentages[i])
            filename = self.audios_names_list[i]
            print(filename)
            plt.ylabel("probability")
            plt.title(filename + " " + model[:12])
            plt.bar(x, y, width=0.6, align='center')
            plt.ylim(0, 1)
            if len(genres[i]) == 1:
                plt.xlim(-1, 1)
            plt.savefig(fname=os.path.join(self.audios_dir, "genre_predictions", filename) + ".png", format='png',
                        bbox_inches="tight", transparent=True)
        plt.clf()

    # endregion audio classification functions

    """
    -----------------------------------------------------------------------------------------------------------
                                        LYRICS CLASSIFICATION FUNCTIONS
    -----------------------------------------------------------------------------------------------------------
    """
    # region lyrics classification functions
    def pressed_lc_radiobutton(self):
        match self.ac_model_selected.get():
            case "Scratch LDA":
                pass
            case "Pretrained LDA":
                pass
    # endregion lyrics classification functions

    """
    -----------------------------------------------------------------------------------------------------------
                                            MUSIC PLAYER FUNCTIONS
    -----------------------------------------------------------------------------------------------------------
    """

    # region music player functions
    def load_audios(self, new_dir_path):
        for root, subdirs, files in os.walk(new_dir_path):
            for filename in files:
                if filename == "vocals.wav" or filename == "accompaniment.wav":
                    continue
                new_filename = os.path.abspath(join(root, filename)).replace("\\", "/")
                if filename.endswith(".wav") and new_filename not in self.audios_paths_list:
                    self.audios_paths_list.append(new_filename)
                    self.audios_names_list.append(filename[:-4])
        if self.audios_names_list:
            self.songs_combobox.configure(values=self.audios_names_list)
            self.songs_combobox.set(self.audios_names_list[0])
            self.change_lyrics()

    def select_file(self):
        from audio_processing import general
        files_list = filedialog.askopenfilenames()
        root_dir = os.path.dirname(__file__)
        new_dir_path = os.path.abspath(join(root_dir, '..', Constants.INPUT_AUDIO))
        for audio in files_list:
            new_audio_path = os.path.normpath(join(new_dir_path, audio.split("/")[-1]))
            if not os.path.normpath(audio) == new_audio_path:
                shutil.copy(audio, new_audio_path)
        general.convert_to_wav(new_dir_path)
        self.load_audios(new_dir_path)

    def handle_recording(self):
        if self.recording == 0:
            self.recording = 1
            # self.recorder.setMicrophone()
            t1 = Thread(target=self.register_audio, daemon=True)
            t1.start()
            # self.register_audio()
            self.register_button.configure(image=self.stop)
        else:
            self.stop_recording()
            self.recording = 0
            self.register_button.configure(image=self.play)
            self.load_audios(Constants.INPUT_AUDIO)

    def register_audio(self):
        self.recorder.start_record()

    def stop_recording(self):
        self.recorder.stop_recording()

    def update_progressbar(self):
        if self.audios_paths_list:
            a = pygame.mixer.Sound(f'{self.audios_paths_list[self.n]}')
            song_len = a.get_length() * 1000
            bar_pos = pygame.mixer.music.get_pos()
            if bar_pos < song_len:
                self.progressbar.set(bar_pos / song_len)
        self.after(1000, self.update_progressbar)

    def play_music(self):
        if self.audios_paths_list:
            self.play_button.configure(image=self.stop)
            song_name = self.audios_paths_list[self.n]
            pygame.mixer.music.load(song_name)
            pygame.mixer.music.play(loops=0)
            pygame.mixer.music.set_volume(.5)

    def create_transcriptions(self):
        from audio_processing import transcription
        input_path = os.path.abspath(join(os.getcwd(), '..', Constants.INPUT_AUDIO))
        # source_separation.source_separation(input_path)
        transcription.fast_transcript(input_path)

    def transcription_threading(self):
        self.lyrics_textbox.insert("0.0", "Loading")
        t1 = Thread(target=self.create_transcriptions, daemon=True)
        t1.start()
        self.monitor(t1)

    def monitor(self, transcription_thread):
        if transcription_thread.is_alive():
            if len(self.lyrics_textbox.get('0.0', customtkinter.END)) > 10:
                self.lyrics_textbox.delete('0.0', customtkinter.END)
                self.lyrics_textbox.insert("0.0", "Loading")
            else:
                self.lyrics_textbox.insert(customtkinter.END, ".")
            self.after(500, lambda: self.monitor(transcription_thread))
        else:
            self.change_lyrics()

    def change_lyrics(self):
        textfile = self.audios_paths_list[self.n][:-4] + ".txt"
        if os.path.exists(textfile):
            with open(textfile) as f:
                lyrics = f.read()
            if lyrics.startswith("  "):
                lyrics = lyrics[2:]
            self.lyrics_textbox.delete('0.0', customtkinter.END)
            self.lyrics_textbox.insert("0.0", lyrics.replace("  ", "\n"))
        else:
            self.lyrics_textbox.delete('0.0', customtkinter.END)

    """def play_vocals(self):
        print()
        # pygame.mixer.music.load(join(Constants.INPUT_AUDIO,self.audios_names_list[self.n], "vocals.wav"))
        # pygame.mixer.music.play(loops=0)
        # pygame.mixer.music.set_volume(.5)

    def play_accompaniment(self):
        pygame.mixer.music.load(join(Constants.INPUT_AUDIO, self.audios_names_list[self.n], "accompaniment.wav"))
        pygame.mixer.music.play(loops=0)
        pygame.mixer.music.set_volume(.5)"""

    def stop_music(self):
        self.stopped = 1
        pygame.mixer.music.pause()

    def handle_player(self):
        if not self.audios_paths_list:
            return
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
        if not self.audios_paths_list:
            return
        self.stopped = 0
        self.n += 1
        if self.n >= len(self.audios_paths_list):
            self.n = 0
        self.play_music()
        self.songs_combobox.set(self.audios_names_list[self.n])
        self.change_lyrics()

    def skip_back(self):
        if not self.audios_paths_list:
            return
        self.stopped = 0
        self.n -= 1
        if self.n < 0:
            self.n = len(self.audios_paths_list) - 1
        self.play_music()
        self.songs_combobox.set(self.audios_names_list[self.n])
        self.change_lyrics()

    def volume(self, value):
        # print(value) # If you care to see the volume value in the terminal, un-comment this :)
        pygame.mixer.music.set_volume(value)

    def change_song(self, choice):
        if not self.audios_names_list:
            return
        self.n = self.audios_names_list.index(choice)
        self.stopped = 0
        self.change_lyrics()
        self.play_music()

    def show_player(self):
        self.play_button.configure(state="normal")
        self.skip_b.configure(state="normal")
        self.skip_f.configure(state="normal")
        self.slider.configure(state="normal")

    # endregion music player functions


if __name__ == "__main__":
    self = App()
    self.mainloop()
