# include all the data processing I need in order to make the project
import audio_processing as ap
import tde_model as tde
import matplotlib.pyplot as plt
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
finality = np.array([])
for file in metadata_tau:
    meta = metadata_tau_loc + file
    meta_csv = pd.read_csv(meta)
    print(meta_csv)
    j = 0  # counter for sound events
    for event in range(len(meta_csv)):
        data = meta_csv.to_numpy()[event]  # keep metadata handy

        j += 1
        spikes_filename = f'/{rev[i]}-{j}-LIF-Response.csv'
        zeros_filename = f'/{rev[i]}-{j}-Zero-Crossings.csv'

        spikers = pd.read_csv(results + spikes_filename).to_numpy().transpose()
        zeros = pd.read_csv(results + zeros_filename).to_numpy().transpose()
        zeros_x = [zeros[1], zeros[2]]
        spk = spikers[2]

        # determine average spikes per burst
        tiempo = []
        for entry in range(len(spk)):
            if spk[entry]:
                tiempo.append(entry)
        isi = np.diff(tiempo)
        isifil = np.array([k for k, g in groupby(isi)])  # remove all repeat consecutive
        peaks = signal.argrelextrema(isifil, np.greater)  # count local maxima
        diff = len(peaks[0])  # gaps btw groups therefore total groups = diff + 1

        avg_spk = torch.sum(spk) / (diff + 1)
        data += avg_spk  # the final results, all metadata including average spiking
        finality.append(data) # add to final array of results
    i += 1

