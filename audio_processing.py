import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import snntorch as snn
import torch
import os
import os.path as path
from matplotlib import colors as mcolors
import pandas as pd
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.pyplot import cm# transfer zeros into spike train
import scipy
from scipy import fft
from scipy import signal
import scipy.io
from scipy.io import wavfile
from scipy import stats
import tde_model as tde
from tde_model import *


# generate SIM data set
def generate_test_waves(angle, frequency, sr, time, distance):

    phase = np.cos(np.deg2rad(angle)) * distance / 343
    t = np.linspace(0, time, time*sr)

    radius = np.array([np.cos(angle), np.sin(angle), 0])
    left = radius + np.array([distance/2, 0, 0])
    right = radius + np.array([-distance/2, 0, 0])

    amp_left = 1 / left.dot(left)
    amp_right = 1 / right.dot(right)

    # start creating outputs
    itd = phase
    ild = amp_left - amp_right
    waveform_1 = amp_left * np.sin(t * frequency)  # right channel
    waveform_2 = amp_right * np.sin((t - phase) * frequency)  # left channel

    return waveform_1, waveform_2, itd, ild


# filtering of real data sound waves
def filter_waves(sig, wc, band):
    """
    Apply butterworth digital signal filter on soundwaves
    :param sig: audio file to be filtered
    :param sr: sampling rate of the audio
    :param band: default low-pass, but can change to high-pass 'hp' or bandpass 'bp'
    :return:
    """
    sos = signal.butter(10, wc, fs=48000, btype=band, output='sos')

    left = sig[0]
    right = sig[1]

    filtered_left = signal.sosfilt(sos, left)
    filtered_right = signal.sosfilt(sos, right)

    waves = [filtered_left, filtered_right]
    return waves


# count zeros and maxima
def zero_crossing(array, sr):
    """
    Encode spike data by writing down zero crossings
    :param array: data containing audio information of each channel
    :param sr: sampling rate used to convert index number into timeslot
    :return spike_data, spike_values: x and y values of crossing
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
    :return spike_data, spike_values: x and y values of each spike location
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

# slice audio file to detect the exact sound event you're looking for


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
    :return x, y: original x and y, now properly snipped for the recording
    """

    # define start and stop array masks, based on time values greater than start time and less than end time
    # use only one array such that the final arrays still correspond
    start_index = x_values[0] > start
    stop_index = x_values[0] < stop

    # apply boolean slices onto the x values and y values
    xf = x_values[0][np.logical_and(start_index, stop_index)]
    yf = y_values[0][np.logical_and(start_index, stop_index)]
    xt = x_values[1][np.logical_and(start_index, stop_index)]
    yt = y_values[1][np.logical_and(start_index, stop_index)]

    x = [xf, xt]
    y = [yf, yt]

    return x, y


# calculate angles from binaural cues
def angle_itd(distance, time, speed=343):
    """
    Given a certain time difference, calculates the angle corresponding to the DoA
    :param distance: interaural distance of the microphones
    :param time: recorded interaural time difference
    :param speed: the speed of sound, set to 343
    :return: angle, given by formula
    """
    return np.arccos(time * speed / distance)


def angle_ild(distance, amplitude):
    """
    Given the corresponding interaural level difference, returns the angle corresponding to the DoA
    :param distance: interaural distance of the microphones
    :param amplitude: recorded interaural level difference
    :return: angle, given by formula
    """
    return np.rad2deg(np.arcsin(2 * amplitude / distance))

# defunct atm

# def count_peak(array):
#     """
#     Pass multichannel audiofile dataset through and return array of spikes
#     indicating which channel produced the highest intensity at given sample.
#
#     :param array: a NxM array of N channels and M samples giving audio intensity
#     :return: spike_numbers, a NxM array of N channels and M samples giving
#     boolean value for spike
#     """
#     spike_numbers = np.zeros([len(array), len(array.transpose())])
#
#     # send a signal each time one microphone detects the louder audio
#     for i in range(len(array.transpose())):  # 'i' is the interval of each timestep
#         peak = np.max(array.transpose()[i])
#         for j in range(len(array)):  # 'j' is the channel source
#             if array[j][i] == peak:
#                 spike_numbers[j][i] = 1
#
#     return spike_numbers
