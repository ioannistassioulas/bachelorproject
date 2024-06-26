import audio_processing as ap
import tde_model as tde
import matplotlib.pyplot as plt

import os
from os import path

from scipy.io import wavfile
from scipy import signal
from scipy import fft

import numpy as np
import pandas as pd
import torch

import time as t
import gc

# start recording time
start_time = t.time()

# get current directory
home = os.getcwd()
one_up = path.abspath(path.join(home, ".."))
metadata_tau_loc = one_up + "/metadata_dev/"
audio_tau_loc = one_up + "/mic_dev/"

# retrieve all filenames
metadata_tau = os.listdir(metadata_tau_loc)
print(f"Acquiring sound files done! Time elapsed = {t.time() - start_time}s")

# array of all files most important info
# setup of array : [audio file, sound_event, metadata of event]
keypoints = []
sr = 24000  # see table on report for justification

# save everything to results file from now on
home = home + '/results/'
n = 0
# start parsing through csv file and saving information
for file in metadata_tau:  # parse through each audio event
    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    audio = audio_tau_loc + root_name + ".wav"

    # process audio file and record to other array
    dummy, intensities = wavfile.read(audio)
    q = 2  # decimation factor

    intensitiesBefore = np.array([intensities.transpose()[0], intensities.transpose()[2]])
    intensitiesAfter = np.array([signal.decimate(intensitiesBefore[0], q), signal.decimate(intensitiesBefore[1], q)])

    a = {"IBL": intensitiesBefore[0], "IBR": intensitiesBefore[1],
                              "IAL": intensitiesAfter[0], "IAR": intensitiesAfter[1]}
    stereoCSV = pd.DataFrame.from_dict(a, orient='index').transpose()
    stereoCSV.to_csv(home + f"Down-Sampling-{n+1}.csv")

    # start reading through specific csv file
    meta_csv = pd.read_csv(meta)

    # create recording arrays to extract information from csv (record of all sound events in file)
    start = []
    end = []
    angle = []
    distance = []
    elevation = []

    for i in meta_csv.transpose():  # parse through each sound event
        sound = np.array(meta_csv.loc[i])

        # append all relevant information
        start.append(sound[1])
        end.append(sound[2])
        elevation.append(sound[3])
        angle.append(sound[4])
        distance.append(sound[5])

    # add onto keypoints mega-array
    file_info = [start, end, elevation, angle, distance]
    keypoints.append(np.array(file_info).transpose())
    n += 1
# print complete message to timestamp
print(f"Reading data done! Time elapsed = {t.time() - start_time}s")

kounter = 0
# START LOOPING THROUGH FILES AND EVENTS
for i in range(len(metadata_tau)):  # start looking at each .wav file

    root_name = metadata_tau[i].partition('.')[0]
    stereo = pd.read_csv(home + f"/Down-Sampling-{i+1}.csv")

    sig = [stereo['IAL'].to_numpy(), stereo['IAR'].to_numpy()]
    sig = [sig[0][~np.isnan(sig[0])], sig[1][~np.isnan(sig[1])]]

    for j in range(len(keypoints[i])):  # go through all sound events
        start = keypoints[i][j][0]
        end = np.min([keypoints[i][j][1], start + 0.5])
        elevation = keypoints[i][j][2]
        angle = keypoints[i][j][3]
        distance = keypoints[i][j][4]

        timespan_len = int((end - start) * sr)

        # Practice filtering out all the data
        timespan = [np.arange(timespan_len), np.arange(timespan_len)]
        timespan, waves = ap.fix_broadcasting(timespan, sig)

        # # To determine frequency band, FFT and find the strongest frequency peaks
        # wave_fft = fft.fft(waves)  # peaks of fft transform
        # freq_fft = fft.fftfreq(len(timespan[0]), 1 / sr)  # frequencies to check over
        # plt.plot(freq_fft, wave_fft[0])
        # plt.plot()
        # plt.show()
        # l_max = np.max(wave_fft[0])
        # r_max = np.max(wave_fft[1])
        # main_freq_l = np.abs(freq_fft[wave_fft[0].argmax()])  # main frequency
        # main_freq_r = np.abs(freq_fft[wave_fft[1].argmax()])  # main frequency

        # b = {"Freq-Range": freq_fft, "Left-FFT": wave_fft[0], "Right-FFT": wave_fft[1]}

        # freq_info = pd.DataFrame.from_dict(b, orient="index").transpose()
        # freq_info.to_csv(home + f"{i+1}-{j+1}-Fourier-Results.csv")
        # avg_freq = int(np.mean([main_freq_l, main_freq_r]))
        # print(f"freq = {avg_freq}")

        freq_band = [495, 505]
        waves = ap.filter_waves(waves, freq_band, "bandpass")
        print(f"Filtering complete! Time elapsed = {t.time() - start_time}s")

        # calculate all zero crossings
        zero_x, zero_y = ap.zero_crossing(waves, sr)
        zero_x[0] = zero_x[0] + start  # fix problems with zeros and linmes crossing on the bar
        zero_x[1] = zero_x[1] + start
        zero_x, zero_y = ap.fix_broadcasting(zero_x, zero_y)

        c = {"ZL": zero_x[0], "ZR": zero_x[1], "WL": waves[0], "WR": waves[1]}
        zero_data = pd.DataFrame.from_dict(c, orient="index").transpose()
        zero_data.to_csv(home + f"{i+1}-{j+1}-Zero-Crossings.csv")
        print(f"Zero crossing complete! Time elapsed = {t.time() - start_time}s")

        fig, ax = plt.subplots(2)
        ax[0].plot(np.arange(len(sig[0][:timespan_len])), sig[0][:timespan_len], label="left")
        ax[0].plot(np.arange(len(sig[1][:timespan_len])), sig[1][:timespan_len], label="right")
        ax[0].set_title("Unfiltered")
        ax[0].legend()
        zero_x[0] = (zero_x[0] - start) * sr  # subtract by start time such that length is consistent with timestep length
        zero_x[1] = (zero_x[1] - start) * sr
        ax[1].plot(np.arange(len(waves[0])), waves[0], color='red', label="left")
        ax[1].plot(np.arange(len(waves[1])), waves[1], color='blue', label="right")
        ax[1].scatter(zero_x[0], [0] * len(zero_x[0]), color='red')
        ax[1].scatter(zero_x[1], [0] * len(zero_x[1]), color='blue')
        ax[1].set_title("Filtered Sound Data")
        ax[1].legend()
        fig.suptitle(f"Filtered vs Unfiltered sound data. Band gap of {freq_band} Hz")
        fig.text(0.5, 0.04, 'Time', ha='center')
        fig.text(0.04, 0.5, 'Intensity', va='center', rotation='vertical')
        plt.show()

        # throw away to try and save a bit of memory
        gc.collect()

        # transfer zeros into spike train
        current_f = torch.zeros(len(timespan[0]))
        for j1 in zero_x[0]:
            current_f[int(j1)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        current_t = torch.zeros(len(timespan[1]))
        for j2 in zero_x[1]:
            current_t[int(j2)] = torch.ones(1)
        # pass spike train into tde simulator

        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.005)
        timer = int((end - start) * sr)
        print("DONE!")

        mem, spk, fac, trg = tde.tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(timer - 1), current_f, current_t)
        d = {"Mem": mem[0], "Spk": spk[0], "Fac": fac[0], "Trg": trg[0]}
        tde_results = pd.DataFrame.from_dict(d, orient="index").transpose()
        tde_results.to_csv(home + f"{i+1}-{j+1}-LIF-Response.csv")
        print(f"TDE simulation complete! TIme elapsed: {t.time() - start_time}s")

        # start counting all spikes
        total_spike = torch.tensor(signal.convolve(spk[0], mem[0]))
        spike_count = torch.sum(total_spike)

        # add spike count to final counter alongside the required angle
        angle = 45 - angle
        results = pd.DataFrame([spike_count, angle, elevation, distance])
        results.to_csv(home + f"{i+1}-{j+1}-Final-Results.csv")
        print(f"Spike encoding complete! TIme elapsed: {t.time() - start_time}s")
        print('DONE!')
        # throw away to try and save a bit of memory
        print(f"{kounter/(len(metadata_tau) * len(keypoints[i]))*100}% Time elapse = {t.time() - start_time}s")

        kounter += 1
        gc.collect()

