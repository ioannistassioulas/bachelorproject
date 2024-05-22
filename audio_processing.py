import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import snntorch as snn
import torch
import os
from matplotlib import colors as mcolors
import pandas as pd
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.pyplot import cm


# use scipy for this part
def filter_waves(freq_threshold, audio, sr):
    """
    Filter soundwave by desired frequency
    :param freq: desired frequency to isolate
    :param audio: audio file to be filtered
    :param sr: sampling rate of the audio
    :return:
    """
    waves = 0
    return waves


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
        derivative = np.gradient(channel)

        # define sign change counting algorithm
        sign = np.sign(derivative)
        current = sign[0]

        for j in range(len(sign)):
            if sign[j] != current: # if the sign changes, update current sign
                if current > 0:  # count only positive peaks
                    x_spike = np.append(x_spike, (j / sr))
                    y_spike = np.append(y_spike, channel[j])
                current = sign[j]

        # read the x_spike and y_spike information into the big array
        spike_data[i] = x_spike
        spike_values[i] = y_spike

    return spike_data, spike_values


def fix_broadcasting(x_values, y_values):
    """
    :param x_values: Two channel array of all x-values in the "encoded" dataset
    :param y_values: Two channel array of all y-values in the "encoded" dataset
    :return facilitatory, trigger: the 2 waves, properly broadcast together
    """

    # unpack values into arrays to check
    xf = x_values[0]
    xt = x_values[1]
    yf = y_values[0]
    yt = y_values[1]

    # sort x_values on basis of which is longer than the other
    length = sorted([xf, xt], key=len)

    # save the final index of the shorter array
    end_point = len(length[0])

    # slice arrays of both waves by the length of the shorter array

    # slice x values
    xf = xf[:end_point]
    xt = xt[:end_point]

    # slice y values
    yf = yf[:end_point]
    yt = yt[:end_point]

    # package first and second wave together
    x_values = [xf, xt]
    y_values = [yf, yt]

    return x_values, y_values


def set_recording(x_values, y_values, start, stop):
    """
    Define time in which to start and stop recording the waves
    :param x_values: 2xN array of x_values of spike information, first and second waves
    :param y_values: 2xN array of y_values of spike information, first and second waves
    :param start: value of time to start recording
    :param stop: value of time to end recording
    :return x_values, y_values: original x and y, now properly snipped for the recording
    """

    # define start and stop array masks, based on time values greater than start time and less than end time
    # use only one array such that the final arrays still correspond
    start_index = x_values[0] > start
    stop_index = x_values[0] < stop

    # apply boolean slices onto the x values and y values
    x_values[0] = x_values[0][np.logical_and(start_index, stop_index)]
    y_values[0] = y_values[0][np.logical_and(start_index, stop_index)]
    x_values[1] = x_values[1][np.logical_and(start_index, stop_index)]
    y_values[1] = y_values[1][np.logical_and(start_index, stop_index)]

    return x_values, y_values


def angle_itd(distance, time, speed=343):
    """
    Given a certain time difference, calculates the angle corresponding to the DoA
    :param distance: interaural distance of the microphones
    :param time: recorded interaural time difference
    :param speed: the speed of sound, set to 343
    :return: angle, given by formula
    """
    return np.rad2deg(np.arccos(time * speed / distance))


def angle_ild(distance, amplitude):
    """
    Given the corresponding interaural level difference, returns the angle corresponding to the DoA
    :param distance: interaural distance of the microphones
    :param amplitude: recorded interaural level difference
    :return: angle, given by formula
    """
    return np.rad2deg(np.arccos(0.5 * amplitude / distance))
