import pyaudio
import wave
import keyboard
import time
from Config import Constants


class Recorder:

    def __init__(self, audio=pyaudio.PyAudio(), device_index=1, chunk=1024, audio_format=pyaudio.paInt16,
                 channels=1, rate=44100, wave_output_filename=Constants.INPUT_AUDIO + "mic_input.wav"):
        # RECORD_SECONDS = 5
        self.audio = audio
        self.chunk = chunk
        self.rate = rate
        self.channels = channels
        self.format = audio_format
        self.device_index = device_index
        self.wave_output_filename = wave_output_filename

    def setMicrophone(self):
        print("----------------------record device list---------------------")
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                      self.audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("-------------------------------------------------------------")
        self.device_index = int(input())

    def record(self):
        stream = self.audio.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 input_device_index=self.device_index,
                                 frames_per_buffer=self.chunk)

        frames = []
        print("Press space to start recording")
        keyboard.wait('space')
        print("Recording...press space to stop")
        time.sleep(0.2)

        # version with keyboard input
        while True:
            try:
                data = stream.read(self.chunk)
                frames.append(data)
            except KeyboardInterrupt:
                break
            if keyboard.is_pressed('space'):
                print("Stopping record after brief delay")
                time.sleep(0.2)
                break

        # version with preset seconds
        """for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("* done recording")"""

        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        wf = wave.open(self.wave_output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()



