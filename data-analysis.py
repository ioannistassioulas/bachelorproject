# include all the data processing I need in order to make the project
import audio_processing as ap
import tde_model as tde
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from itertools import groupby


import os
from os import path
import glob

from scipy.io import wavfile
from scipy import signal
from scipy import fft

import numpy as np
import pandas as pd
import torch

import time as t
import gc
import re
time_start  = t.time()
# read in all the result csv files
home = os.getcwd()
one_up = path.abspath(path.join(home, ".."))
metadata_tau_loc = one_up + "/metadata_dev/"
metadata_tau = os.listdir(metadata_tau_loc)
metadata_tau.sort()
rev = []

# save number for timeslot
for i in metadata_tau:
    reversi = i[::-1]
    numb = reversi.partition('.')[2].partition('_')[0]
    fin = numb[::-1]
    rev.append(fin)

results = os.getcwd() + "/results"

i = 0  # counter for sound files
finality = np.zeros(6)

for file in metadata_tau:
    meta = metadata_tau_loc + file
    meta_csv = pd.read_csv(meta)
    j = 0  # counter for sound events
    for event in range(len(meta_csv)):
        data = meta_csv.to_numpy()[event][1:]  # keep metadata handy
        j += 1
        spikes_filename = f'/{rev[i]}-{j}-LIF-Response.csv'
        zeros_filename = f'/{rev[i]}-{j}-Zero-Crossings.csv'

        spikers = pd.read_csv(results + spikes_filename).to_numpy().transpose()
        zeros = pd.read_csv(results + zeros_filename).to_numpy().transpose()
        zeros_x = [zeros[1], zeros[2]]
        spk = spikers[2]
        spk = [int(re.findall('[0-9]+', i)[0]) for i in spk]

        # determine average spikes per burst
        tiempo = []
        for entry in range(len(spk)):
            if spk[entry]:
                tiempo.append(entry)

        isi = np.diff(tiempo)
        isifil = np.array([k for k, g in groupby(isi)])  # remove all repeat consecutive

        peaks = signal.argrelextrema(isifil, np.greater)  # count local maxima
        diff = len(peaks[0])  # gaps btw groups therefore total groups = diff + 1

        avg_spk = [np.sum(spk) / (diff + 1)]
        data = np.concatenate((data, avg_spk)) # the final results, all metadata including average spiking
        finality = np.vstack((finality, data))  # add to final array of results
    i += 1
finality = finality[1:]


# sort based on distance
one_meter = []
two_meter = []
for i in finality:
    if i[4] == 1:
        one_meter.append(i)
    else:
        two_meter.append(i)
# sort based on elevation

one_meter_dic = {
    40 : [],
    30 : [],
    20 : [],
    10 : [],
     0 : [],
   -10 : [],
   -20 : [],
   -30 : [],
   -40 : [],
}

two_meter_dic = {
    20  : [],
    10  : [],
    0   : [],
    -10 : [],
    -20 : []
}

for input in one_meter:
    one_meter_dic[input[2]].append((input[3], input[5]))
print("done!")
for input2 in two_meter:
    two_meter_dic[input2[2]].append((input2[3], input2[5]))

print(f"Only took {t.time() - time_start}s !!")
# create graphs for both 1 meter and 2 meter

# 1 meter
color = iter(cm.rainbow(np.linspace(0, 1, 10)))

fit_value_by_elevation = []
for i in one_meter_dic:  # go by elevation
    res = np.array(one_meter_dic[i])

    theta = res.transpose()[0]
    spe = res.transpose()[1]

    res_dict = {
        90: [],
        80: [],
        70: [],
        60: [],
        50: [],
        40: [],
        30: [],
        20: [],
        10: [],
         0: []
    }

    start_angle = theta <= 90
    end_angle = theta >= 0

    theta = theta[np.logical_and(start_angle, end_angle)]
    spe = spe[np.logical_and(start_angle, end_angle)]

    for entry in range(len(theta)):
        res_dict[theta[entry]].append(spe[entry])

    angles = list(res_dict.keys())
    spe_avg = [np.mean(np.array(element)) for element in res_dict.values()]
    spe_err = [np.std(np.array(element)) for element in res_dict.values()]

    p, V = np.polyfit(angles, spe_avg, 1, cov=True)

    cor = next(color)
    plt.scatter(angles, spe_avg, label=f"Elevation = {i}", c=cor)
    plt.errorbar(angles, spe_avg, spe_err, c=cor, linestyle='')
    plt.plot(angles, p[0] * np.array(angles) + p[1], c=cor)
    plt.xlabel(r"Angles [$^{\circ}$]")
    plt.ylabel("Average Spikes per burst")
    plt.title("")
    plt.legend()
    plt.show()

    fit_value_by_elevation.append((p[0], V[0][0]))

fit_value = np.array(fit_value_by_elevation).transpose()
print(fit_value)
plt.scatter(list(one_meter_dic.keys()), fit_value[0])
plt.errorbar(list(one_meter_dic.keys()), fit_value[0], np.sqrt(fit_value[1]), linestyle='')
plt.xlabel("Elevation")
plt.ylabel("Slope of Linear Fit")
plt.title("Sound event from 1 meters")
plt.show()
# 2 meter

fit_value_by_elevation = []
color = iter(cm.rainbow(np.linspace(0, 1, 10)))
for i in two_meter_dic:  # go by elevation
    res = np.array(two_meter_dic[i])

    theta = res.transpose()[0]
    spe = res.transpose()[1]

    res_dict = {
        90: [],
        80: [],
        70: [],
        60: [],
        50: [],
        40: [],
        30: [],
        20: [],
        10: [],
         0: []
    }

    start_angle = theta <= 90
    end_angle = theta >= 0

    theta = theta[np.logical_and(start_angle, end_angle)]
    spe = spe[np.logical_and(start_angle, end_angle)]

    for entry in range(len(theta)):
        res_dict[theta[entry]].append(spe[entry])

    angles = list(res_dict.keys())

    spe_avg = [np.mean(np.array(element)) for element in res_dict.values()]
    spe_err = [np.std(np.array(element)) for element in res_dict.values()]
    p, V = np.polyfit(angles, spe_avg, 1, cov=True)

    cor = next(color)
    plt.scatter(angles, spe_avg, label=f"Elevation = {i}", c=cor)
    plt.errorbar(angles, spe_avg, spe_err, c=cor, linestyle='')
    plt.plot(angles, p[0] * np.array(angles) + p[1], c=cor)
    plt.xlabel(r"Angles [$^{\circ}$]")
    plt.ylabel("Average Spikes per burst")
    plt.title("")
    plt.legend()
    plt.show()

    fit_value_by_elevation.append((p[0], V[0][0]))

fit_value = np.array(fit_value_by_elevation).transpose()
plt.scatter(list(two_meter_dic.keys()), fit_value[0])
plt.errorbar(list(two_meter_dic.keys()), fit_value[0], np.sqrt(fit_value[1]), linestyle='')

p, V = np.polyfit(list(two_meter_dic.keys()), fit_value[0], 1, cov=True)
plt.plot(list(two_meter_dic.keys()), p[0] * list(two_meter_dic.keys()) + p[1], label = f"{}")
plt.xlabel("Elevation")
plt.ylabel("Slope of Linear Fit")
plt.title("Sound event from 2 meters")
plt.show()