# Based on code from Siriwardhana, C. (2019) Real-time Sound event classification, Medium. Available at: https://towardsdatascience.com/real-time-sound-event-classification-83e892cf187e (Accessed: 1 June 2022).

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyaudio
import wave
from keras.models import model_from_json
import librosa
import serial
from simplecoremidi import send_midi

# Define serial port of ESP32 device
arduino = serial.Serial(port='/dev/tty.SLAB_USBtoUART', baudrate=115200, timeout=.1)

# Load the model and weights
file = open('weights-binary(256).json', 'r')
model = model_from_json(file.read())
model.load_weights('weights-binary(256).h5')

# The initial classification takes longer so the model is initiated with a silent audio file
x = np.zeros(4608)
stft = abs(librosa.stft(x, n_fft=1024, center=False))
X = np.reshape(stft, (1, 513, 15, 1))
result = model.predict(X)

last = 1

def Classifier(x_app):
    global x
    global last
    # Take the last 3 windows
    x = x[3072:4608]
    # and append 12 more recent windows
    x = np.append(x, x_app)
    stft = abs(librosa.stft(x, n_fft=1024, center=False))
    X = np.reshape(stft, (1, 513, 15, 1))
    # Calculate the model prediction
    result = model.predict(X)
    if result[0][0] >= 0.5:
        # If the last result was not an onset
        if last != 0:
            # Debugging
            print('onset')
            arduino.write(bytes('1', 'utf-8'))
            # Send MIDI note on and note off for testing
            send_midi((0x90, 0x3c, 0x40))
            send_midi((0x80, 0x3c, 0x40))
            last = 0
    elif result[0][0] < 0.5:
        last = 1

CHUNKSIZE = 3072  # fixed chunk size
RATE = 44100

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
audio_buffer = []

while True:
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    whole_window = np.frombuffer(data, dtype=np.int16)
    current_window = whole_window[1::2]
    current_window = current_window.astype(np.float32, order='C') / 32768.0
    Classifier(np.array(current_window))
    current_window = []

