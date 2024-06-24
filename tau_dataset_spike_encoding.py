import audio_processing as ap
import tde_model as tde

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
sr = 28000

home = home + '/results'
os.mkdir(home)

i = 0
# start parsing through csv file and saving information
for file in metadata_tau:  # parse through each audio event
    soundwave_file = home + f'/file_{i+1}'
    os.mkdir(soundwave_file)

    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    audio = audio_tau_loc + root_name + ".wav"

    # process audio file and record to other array
    dummy, intensities = wavfile.read(audio)
    samp = int(len(intensities) * sr / dummy)

    intensitiesBefore = np.array([intensities.transpose()[0], intensities.transpose()[2]])
    intensitiesAfter = np.array([signal.resample(intensitiesBefore[0], samp), signal.resample(intensitiesBefore[1], samp)])
    a = {"IBL": intensitiesBefore[0], "IBR": intensitiesBefore[1],
                              "IAL": intensitiesAfter[0], "IAR": intensitiesAfter[1]}
    stereoCSV = pd.DataFrame.from_dict(a, orient='index').transpose()
    stereoCSV.to_csv(soundwave_file + f"/down-sampling.csv")

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
    i += 1
# print complete message to timestamp
print(f"Reading data done! Time elapsed = {t.time() - start_time}s")

# START LOOPING THROUGH FILES AND EVENTS
for i in range(len(metadata_tau)):  # start looking at each .wav file

    root_name = metadata_tau[i].partition('.')[0]
    soundwave_file = home + f'/file_{i + 1}'
    stereo = pd.read_csv(soundwave_file + f"/down-sampling.csv")
    sig = [stereo['IAL'].to_numpy(), stereo['IAR'].to_numpy()]
    sig = [sig[0][~np.isnan(sig[0])], sig[1][~np.isnan(sig[1])]]

    for j in range(1):  # go through all sound events
        loc = soundwave_file + f'/event_{j+1}'
        os.mkdir(loc)
        start = keypoints[i][j][0]
        end = keypoints[i][j][1]
        elevation = keypoints[i][j][2]
        angle = keypoints[i][j][3]
        distance = keypoints[i][j][4]

        # Practice filtering out all the data
        timespan = [np.arange(len(np.array(sig[0]))), np.arange(len(np.array(sig[1])))]

        # look at events
        timespan, waves = ap.fix_broadcasting(timespan, sig)
        timespan, waves = ap.set_recording(timespan, waves, int(start * sr),
                                                          int(end * sr))

        unfiltered = waves
        time_unfiltered = timespan

        # To determine frequency band, FFT and find the strongest frequency peaks
        wave_fft = fft.fft(waves)  # peaks of fft transform
        freq_fft = fft.fftfreq(len(timespan[0]), 1 / sr)  # frequencies to check over

        l_max = np.max(wave_fft[0])
        r_max = np.max(wave_fft[1])
        main_freq_l = np.abs(freq_fft[wave_fft[0].argmax()])  # main frequency
        main_freq_r = np.abs(freq_fft[wave_fft[1].argmax()])  # main frequency

        b = {"Freq-Range": freq_fft, "Left-FFT": wave_fft[0], "Right-FFT": wave_fft[1]}

        freq_info = pd.DataFrame.from_dict(b, orient="index").transpose()
        freq_info.to_csv(loc + f"/Fourier-Results.csv")
        avg_freq = int(np.mean([main_freq_l, main_freq_r]))
        freq_band = [avg_freq - 5, avg_freq + 5]

        waves = ap.filter_waves(waves, freq_band, "bandpass")
        print(f"Filtering complete! Time elapsed = {t.time() - start_time}s")

        # calculate all zero crossings
        zero_x, zero_y = ap.zero_crossing(waves, sr)
        zero_x[0] = zero_x[0] + start  # fix problems with zeros and linmes crossing on the bar
        zero_x[1] = zero_x[1] + start
        zero_x, zero_y = ap.fix_broadcasting(zero_x, zero_y)

        c = {"ZL": zero_x[0], "ZR": zero_x[1], "WL": waves[0], "WR": waves[1]}
        zero_data = pd.DataFrame.from_dict(c, orient="index").transpose()
        zero_data.to_csv(loc + f"/Zero-Crossings.csv")
        print(f"Zero crossing complete! Time elapsed = {t.time() - start_time}s")

        # delete all time intervals that are higher than expected (due to fault)
        time_differences = np.abs(zero_x[1] - zero_x[0])  # experimental phase crossing
        inter_distance = 2 * np.sin(np.deg2rad(35)) * 0.042
        final_angle = np.rad2deg(ap.angle_itd(inter_distance, time_differences))

        # throw away to try and save a bit of memory
        gc.collect()

        # transfer zeros into spike train
        zero_x[0] = (zero_x[0] - start) * sr  # subtract by start time such that length is consistent with timestep length
        current_f = torch.zeros(len(timespan[0]))
        for j1 in zero_x[0]:
            current_f[int(j1)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        # repeating for trigger input
        zero_x[1] = (zero_x[1] - start) * sr
        current_t = torch.zeros(len(timespan[1]))
        for j2 in zero_x[1]:
            current_t[int(j2)] = torch.ones(1)

        # pass spike train into tde simulator

        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.005)
        timer = int((end - start) * sr)

        mem, spk, fac, trg = tde.tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(timer - 1), current_f, current_t)
        d = {"Mem": mem[0], "Spk": spk[0], "Fac": fac[0], "Trg": trg[0]}
        tde_results = pd.DataFrame.from_dict(d, orient="index").transpose()
        tde_results.to_csv(loc + f"/LIF-Response.csv")
        print(f"TDE simulation complete! TIme elapsed: {t.time() - start_time}s")
        # start counting all spikes
        total_spike = torch.tensor(signal.convolve(spk[0], mem[0]))
        spike_count = torch.sum(total_spike)

        # add spike count to final counter alongside the required angle
        angle = 45 - angle
        results = pd.DataFrame([spike_count, angle, elevation, distance])
        results.to_csv(loc + f"/Final-Results.csv")
        print(f"Spike encoding complete! TIme elapsed: {t.time() - start_time}s")

        # throw away to try and save a bit of memory
        gc.collect()

