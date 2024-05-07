# Figure out how to use Torch properly to create your own neural network
import snntorch.spikegen

import audio_processing
from audio_processing import *

# metadata involves array of all info in the order of angles, frequency to parse through in
# the function calls, and then ground truth time difference to compare,
# followed  by finally the calculated time difference from the simulations
metadata = np.array([[15, 30, 45, 60, 75], [200, 500, 800, 2000, 5000, 8000],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372],
                     np.zeros(5*6)], dtype=object)

home = os.getcwd()
database = home + "/datasets/synthetic_data"

for i in range(len(metadata[0])):
    for j in range(len(metadata[1])):
        # download sound files and separate into spike datafile
        loc = audio_processing.name_parse(database, metadata[0][i], metadata[1][j])
        stereo, sampling_rate = librosa.load(loc, mono=False)
        spike_data, spike_values = audio_processing.zero_crossing(stereo, sampling_rate, 0.05)  # generate spike data values

        # fix any broadcasting issues
        length = sorted([spike_data[0], spike_data[1]], key=len)
        spike_data[0] = spike_data[0][:len(length[0])]
        spike_data[1] = spike_data[1][:len(length[0])]

        # determine the inter aural time difference from the data amassed
        time_difference = np.abs(spike_data[1] - spike_data[0])  # find difference in zero crossings from both channels
        metadata[3][(i*6) + j] = np.mean(time_difference)
        # print(f"angle is {np.rad2deg(np.arccos(343 * np.mean(time_difference)/0.3))} for angle {metadata[0][i]} and frequency {metadata[1][j]}")# encode experimental data
        # itd_experimental_error = np.std(time_difference)

# determine the angle difference as per the derivation from Hugh's paper
angle_mean = np.reshape(np.rad2deg(audio_processing.angle_by_itd(*metadata[3])), [5, 6])
angle_theory = np.rad2deg(audio_processing.angle_by_itd(*metadata[2]))

print(angle_mean)
print(angle_theory)
# plot inter aural time difference alongside the angle in the legend
# comparing performance of different frequencies at different angles

for i in range(5):
    plt.scatter(metadata[1], angle_mean[i], label=f"angle = {metadata[0][i]}")
    plt.axhline(angle_theory[i], label="Ground truth")

    plt.title(f"Performance of zero crossings method for angle of {metadata[0][i]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle")
    plt.legend()
    plt.show()

# time differences to encode while creating sample synthetic data (frequency independent)
# 15 deg itd = 0.000844834 s
# 30 deg itd = 0.000757457 s
# 45 deg itd = 0.000618461 s
# 60 deg itd = 0.000437318 s
# 75 deg itd = 0.000226372 s

