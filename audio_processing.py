# Purpose: this code seeks to transform a .wav file into an array of datapoints that can be plotted into a waves
# Upon accomplishing this, the code will export the datapoints to other parts of the project for the purpose of
# generating spikes

# This is done through multiple difference ways

# To DO:
# 1. upload all used data files into GitHub, such that code is consistent between both locations
# 2. consider including a library file into the project such that I don't need to keep importing from here

import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import snntorch as snn
import torch
import os
from matplotlib import colors as mcolors
import pandas as pd


# import stereo dataset


def count_peak(array):
    """
    Pass multichannel audiofile dataset through and return array of spikes
    indicating which channel produced the highest intensity at given sample.

    :param array: a NxM array of N channels and M samples giving audio intensity
    :return: spike_numbers, a NxM array of N channels and M samples giving
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


def peak_diff(array):
    left_channel = array[0]
    right_channel = array[1]

    return left_channel - right_channel


def count_peak_diff(array):
    """
    Adaption of count_peak() - now, the time difference between is counted
    and spikes are produced based on how much louder one is than another

    :param array: a NxM array of N channels and M samples giving audio intensity
    :return: spike_numbers, a NxM array of N channels and M samples and each entry
    gives a frequency for spike rate
    """

    left_channel = np.zeros(len(array.transpose()))
    right_channel = np.zeros(len(array.transpose()))

    for i in range(len(array.transpose())):
        spike_rate = np.round((array[0][i] - array[1][i]) * 1000)  # spike rate determined by comparison to mean
        if spike_rate >= 0:  # assign spike rate to corresponding spot on spike matrix
            left_channel[i] = np.abs(spike_rate)
        else:
            right_channel[i] = np.abs(spike_rate)

    return left_channel, right_channel


def count_peak_ratio(array):
    """
    Count the ratio of the peak of each array at each timestep (left over right)
    :param array: the 2 channel audio source
    :return: ratio
    """
    left_ear = array[0]
    right_ear = array[1]

    return left_ear / right_ear


def zero_crossing(array, sr):
    """
    Encode spike data by writing down zero crossings
    :param array: data containing audio information of each channel
    :param sr: sampling rate used to convert index number into timeslot
    :param cutoff: cutoff potential to prevent overcounting
    :param stop: ending point such that exact amount is taken
    :return: spike_data, array of timestamps where zero is crossed
    :return: spike_values, array of intensity values where zero crossing is recorded
    """

    spike_data = np.zeros(len(array), dtype=object)
    spike_values = np.zeros(len(array), dtype=object)

    for i in range(len(array)):
        # take first array and record at cutoff index
        channel = array[i]
        sign = np.sign(channel)
        current = sign[0]

        output = np.array([])  # crossing points
        values = np.array([])  # values

        for j in range(len(sign)):
            if sign[j] != current:
                output = np.append(output, (j / sr))
                values = np.append(values, channel[j])
                current = sign[j]
        spike_data[i] = output
        spike_values[i] = values

    return spike_data, spike_values


def peak_difference(array, sr):
    """
    Count peaks of each wave
    :param array: array of 2 channel audio
    :param sr: sampling rate of audio
    :return: spike_data. x values of each spike location
    :return: spike_values, y values of each spike location
    """
    spike_data = np.zeros(len(array), dtype=object)
    spike_values = np.zeros(len(array), dtype=object)

    for i in range(len(array)):
        # initialize all the most important values first
        channel = array[i]
        x_spike = np.array([])
        y_spike = np.array([])

        # create array of derivatives
        derivative = np.zeros(len(channel)-1)
        for j in range(len(derivative)):
            derivative[j] = channel[j] - channel[j+1]

        # define sign change counting algorithm
        sign = np.sign(derivative)
        current = sign[0]
        for j in range(len(sign)):
            if sign[j] != current:
                x_spike = np.append(x_spike, (j / sr))
                y_spike = np.append(y_spike, channel[j])
                current = sign[j]

        # read the x_spike and y_spike information into the big array
        spike_data[i] = x_spike
        spike_values[i] = y_spike

    return spike_data, spike_values

def name_parse(direc, angle, frequency):
    """
    :param direc: destination of dataset files
    :param angle: angle of the specific dataname
    :param frequency: frequency of sound wave
    :return: filename to be appended to end of string
    """
    filename = direc + f"/{angle}_deg_{frequency}_Hz.wav"
    return filename


def angle_by_itd(distance, *time):
    angle = np.zeros(len(time))
    for i in range(len(time)):
        angle[i] = np.arccos(time[i] * 343 / distance)
    return angle


def angle_ild(distance, *amp):
    angle = np.zeros(len(amp))
    for i in range(len(amp)):
        angle[i] = np.arccos(amp[i] / (2 * distance))
    return angle


def display_waves(stereo, sampling_rate):
    """
    print out the waveforms of each of the channels passed through
    Input: array, a NxM array of N channels and M samples giving audio intensity
    Output
    """
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(stereo[0], sr=sampling_rate, label="second source", color="blue")
    librosa.display.waveshow(stereo[1], sr=sampling_rate, label="first source", color="red")
    plt.legend()
    plt.show()

# def cross_correlation(first, second):
#     return np.sum(first * second) *
