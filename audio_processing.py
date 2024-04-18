#Purpose: this code seeks to transform a .wav file into an array of datapoints that can be plotted into a waves
#Upon accomplishing this, the code will export the datapoints to other parts of the project for the purpose of
#generating spikes

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

#import stereo dataset
stereo, sampling_rate = librosa.load("/home/s4290755/Downloads/sample_sound_data.wav", mono=False)


#plot waveform and display on screen
# plt.figure().set_figwidth(12)
# librosa.display.waveshow(array[2], sr=sampling_rate, label = "second source")
# librosa.display.waveshow(array[0], sr=sampling_rate, label = "first source")
# plt.legend()
# plt.show()
#

def count_peak(array):

    spike_numbers = np.zeros([len(array), len(array.transpose())])

    #send a signal each time one microphone detects the louder audio
    for i in range(len(array.transpose())): #i is the interval of each timestep
        peak = np.max(array.transpose()[i])
        for j in range(len(array)): #j is the channel source
            if array[j][i] == peak:
                spike_numbers[j][i] = 1

    return spike_numbers

print(count_peak(stereo))

