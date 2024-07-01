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

results = os.getcwd() + "/results"

i = 0  # counter for sound files
for file in metadata_tau:
    i += 1
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    meta_csv = pd.read_csv(meta)

    j = 0  # counter for sound events
    for event in range(len(meta_csv.tranpose())):
        j += 1
        spikes_filename = f'{i}-{j}-LIF-Response.csv'
        zeros_filename = f'{i}-{j}-Zero-Crossings.csv'

        spikers = pd.read_csv(results + spikes_filename).to_numpy()
        zeros = pd.read_csv(results + zeros_filename).to_numpy().transpose()
        print(zeros)

# code to determine average spikes per burst
spikers = []
tiempo = []
for entry in range(len(spk[0])):
    if spk[0][entry]:
        tiempo.append(entry)
isi = np.diff(tiempo)
isifil = np.array([k for k, g in groupby(isi)])
peaks = signal.argrelextrema(isifil, np.greater)
diff = len(peaks[0])  # gaps btw groups therefore total groups = diff + 1

avg_spk_per_group = torch.sum(spk[0]) / (diff + 1)
result = [avg_spk_per_group, angle]
spikers.append(result)