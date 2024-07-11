import audio_processing as ap
import tde_model as tde
import matplotlib.pyplot as plt
from itertools import groupby


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
# sr = 48000  # see table on report for justification

# save everything to results file from now on
home = home + '/results/'
# start parsing through csv file and saving information
n = 1
for file in metadata_tau:  # parse through each audio event
    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    audio = audio_tau_loc + root_name + ".wav"

    # process audio file and record to other array
    sr, intensities = wavfile.read(audio)
    intensitiesBefore = np.array([intensities.transpose()[0], intensities.transpose()[2]])
    a = {"IBL": intensitiesBefore[0], "IBR": intensitiesBefore[1],
                              "IAL": intensitiesBefore[0], "IAR": intensitiesBefore[1]}
    stereoCSV = pd.DataFrame.from_dict(a, orient='index').transpose()
    stereoCSV.to_csv(home + f"Down-Sampling-{n}.csv")

    # # start reading through specific csv file
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

# START LOOPING THROUGH FILES AND EVENTS
n = 1
for i in range(len(metadata_tau)):  # start looking at each .wav file
    numb = int(metadata_tau[i][::-1].partition('.')[2].partition('_')[0][::-1])  # index of the time to take

    root_name = metadata_tau[i].partition('.')[0]
    stereo = pd.read_csv(home + f"Down-Sampling-{n}.csv")

    sig = [stereo['IAL'].to_numpy(), stereo['IAR'].to_numpy()]
    sig = [sig[0][~np.isnan(sig[0])], sig[1][~np.isnan(sig[1])]]

    for j in range(len(keypoints[i])):  # go through all sound events

        start = keypoints[i][j][0]
        end = keypoints[i][j][1]
        elevation = keypoints[i][j][2]
        angle = keypoints[i][j][3]
        distance = keypoints[i][j][4]

        timespan_len = int((end - start) * sr)

        # Practice filtering out all the data
        timespan = [np.arange(len(sig[0])), np.arange(len(sig[1]))]
        timespan, waves = ap.fix_broadcasting(timespan, sig)
        print(len(waves[0]))
        timespan, waves = ap.set_recording(timespan, waves, start * sr, end * sr)
        # To determine frequency band, FFT and find the strongest frequency peaks

        band_gap = [480, 520]
        wave_fft = ap.filter_waves(waves, band_gap, "bandpass")

        # only measure first 0.1 seconds
        index = int(0.1 * sr)
        wave_fft[0] = wave_fft[0][:index]
        wave_fft[1] = wave_fft[1][:index]

        print(f"Filtering complete! Time elapsed = {t.time() - start_time}s")

        # calculate all zero crossings
        zero_x, zero_y = ap.zero_crossing(wave_fft, sr)
        zero_x[0] = (zero_x[0]) * sr # fix problems with zeros and linmes crossing on the bar
        zero_x[1] = (zero_x[1]) * sr

        zero_x, zero_y = ap.fix_broadcasting(zero_x, zero_y)

        c = {"ZL": zero_x[0], "ZR": zero_x[1], "WL": wave_fft[0], "WR": wave_fft[1]}
        zero_data = pd.DataFrame.from_dict(c, orient="index").transpose()
        zero_data.to_csv(home + f"{numb}-{j+1}-Zero-Crossings.csv")
        print(home + f"{numb}-{j+1}-Zero-Crossings.csv")
        print(f"Zero crossing complete! Time elapsed = {t.time() - start_time}s")



        # throw away to try and save a bit of memory
        gc.collect()

        # transfer zeros into spike train
        current_f = torch.zeros(len(wave_fft[0]))
        print(len(wave_fft[0]))
        for j1 in zero_x[0]:
            current_f[int(j1)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        current_t = torch.zeros(len(wave_fft[1]))
        for j2 in zero_x[1]:
            current_t[int(j2)] = torch.ones(1)

        # pass spike train into tde simulator
        tau_tde = torch.tensor(20/sr)
        tau_mem = torch.tensor(20/sr)

        current_f = current_f[:index]
        current_t = current_t[:index]

        mem, spk, fac, trg = tde.tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(len(current_f)), current_f, current_t)

        tiempo = []
        for entry in range(len(spk[0])):
            if spk[0][entry]:
                tiempo.append(entry)
        isi = np.diff(tiempo)
        isifil = np.array([k for k, g in groupby(isi)])
        peaks = signal.argrelextrema(isifil, np.greater)
        diff = len(peaks[0])  # gaps btw groups therefore total groups = diff + 1

        d = {"Mem": mem[0], "Spk": spk[0], "Fac": fac[0], "Trg": trg[0]}
        tde_results = pd.DataFrame.from_dict(d, orient="index").transpose()
        tde_results.to_csv(home + f"{numb}-{j+1}-LIF-Response.csv")
        print(f"TDE simulation complete! TIme elapsed: {t.time() - start_time}s")

        gc.collect()
    n += 1


