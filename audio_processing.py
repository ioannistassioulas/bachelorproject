# Purpose: this code seeks to transform a .wav file into an array of datapoints that can be plotted into a waves
# Upon accomplishing this, the code will export the datapoints to other parts of the project for the purpose of
# generating spikes

# To DO:
# 1. upload all used data files into GitHub, such that code is consistent between both locations
# 2. consider including a library file into the project such that I don't need to keep importing from here

import numpy as np
import librosa.display

# import stereo dataset
stereo, sampling_rate = librosa.load("/home/s4290755/Downloads/sample_sound_data.wav", mono=False)


def count_peak(array):
    """
    Purpose: pass multichannel audiofile dataset through and return array of spikes
    indicating which channel produced the highest intensity at given sample.

    Input: array, a NxM array of N channels and M samples giving audio intensity
    Output: spike_numbers, a NxM array of N channels and M samples giving
    boolean value for spike
    """
    spike_numbers = np.zeros([len(array), len(array.transpose())])

    # send a signal each time one microphone detects the louder audio
    for i in range(len(array.transpose())):  # 'i' is the interval of each timestep
        peak = np.max(array.transpose()[i])
        for j in range(len(array)):  # 'j' is the channel source
            if array[j][i] == peak:
                spike_numbers[j][i] = 1

    return spike_numbers


print(count_peak(stereo))
