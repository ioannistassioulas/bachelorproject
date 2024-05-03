# Figure out how to use Torch properly to create your own neural network
import snntorch.spikegen

import audio_processing
from pathlib import Path
from audio_processing import *
import snntorch as snn
import torch

# download sound files and separate into spike data
stereo, sampling_rate = librosa.load(, mono=False)
spike_data, spike_values = audio_processing.zero_crossing(stereo, sampling_rate)  # generate spike data values

# metadata involves array of all info in the order of angles, time difference, frequency to parse through in
# the function calls
metadata = np.array([[15, 30, 45, 60, 75],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372],
                     [200, 500, 800, 2000, 5000, 8000]], dtype=object)

# determine the inter aural time difference from the data amassed
time_difference = np.abs(spike_data[1] - spike_data[0])  # find difference in zero crossings from both channels

itd_experimental_mean = np.mean(time_difference)
itd_experimental_error = np.std(time_difference)
itd_theoretical = 0.000770  # read directly from audacity

# determine the angle difference as per the derivation from Hugh's paper
angle_mean = np.rad2deg(audio_processing.angle_by_itd(itd_experimental_mean))
angle_theory = np.rad2deg(audio_processing.angle_by_itd(itd_theoretical))

# plot inter aural time difference alongside the angle in the legend
plt.plot(spike_data[0], time_difference, label=f"experimental data, angle = {angle_mean}")  # measured timestep difference
plt.axhline(itd_theoretical, color="red", label=f"ground truth, angle = {angle_theory}")  # ground truth as per our formula
plt.legend()
plt.show()

# time differences to encode while creating sample synthetic data (frequency independent)
# 15 deg itd = 0.000844834 s
# 30 deg itd = 0.000757457 s
# 45 deg itd = 0.000618461 s
# 60 deg itd = 0.000437318 s
# 75 deg itd = 0.000226372 s

