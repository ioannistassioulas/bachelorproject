from audio_processing import *
import matplotlib.pyplot as plt


# plot waveform and display on screen
def display_waves(array):
    """
    print out the waveforms of each of the channels passed through
    Input: array, a NxM array of N channels and M samples giving audio intensity
    Output
    """
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(array[2], sr=sampling_rate, label="second source")
    librosa.display.waveshow(array[0], sr=sampling_rate, label="first source")
    plt.legend()
    plt.show()
