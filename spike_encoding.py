# Figure out how to use Torch properly to create your own neural network

# time differences to encode while creating sample synthetic data (frequency independent)
# 15 deg itd = 0.000844834 s
# 30 deg itd = 0.000757457 s
# 45 deg itd = 0.000618461 s
# 60 deg itd = 0.000437318 s
# 75 deg itd = 0.000226372 s

import audio_processing
from audio_processing import *

# metadata involves array of all info in the order of angles, frequency to parse through in
# the function calls, and then ground truth time difference to compare,
# followed  by finally the calculated time difference from the simulations and the
# level difference in the simulations
metadata = np.array([[15, 30, 45, 60, 75], [200, 500, 800, 2000, 5000, 8000],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372],
                     np.zeros(5 * 6), np.zeros(5 * 6), ["red", "blue", "orange", "green", "black", "brown"]],
                    dtype=object)

# determine the directory location wrt current working directory
home = os.getcwd()
database = home + "/datasets/synthetic_data"

for i in range(len(metadata[0])):  # parse through all angles
    for j in range(len(metadata[1])):  # parse through all frequencies

        # download sound files and separate into spike datafile
        loc = audio_processing.name_parse(database, metadata[0][i], metadata[1][j])
        stereo, sampling_rate = librosa.load(loc, mono=False)

        # create data for ITD
        spike_data, spike_values = audio_processing.zero_crossing(stereo, sampling_rate,
                                                                  0.001)  # generate spike data values

        # fix any broadcasting issues
        length = sorted([spike_data[0], spike_data[1]], key=len)
        spike_data[0] = spike_data[0][:len(length[0])]
        spike_data[1] = spike_data[1][:len(length[0])]

        # determine the inter aural time difference from the data amassed
        time_difference = np.abs(spike_data[1] - spike_data[0])  # find difference in zero crossings from both channels
        metadata[3][(i * 6) + j] = np.mean(time_difference)  # return mean into metadata super array

        # determine the inter aural level difference from data amassed per sample step
        max_level_difference = np.abs(np.nanmean(audio_processing.count_peak_ratio(stereo)))
        metadata[4][(i * 6) + j] = max_level_difference

# determine the angle difference as per the derivation from Hugh's paper
angle_mean = np.reshape(np.rad2deg(audio_processing.angle_by_itd(*metadata[3])), [5, 6])
angle_theory = np.rad2deg(audio_processing.angle_by_itd(*metadata[2]))

level_mean = np.reshape(metadata[4], [5, 6]).transpose()  # now it goes frequency, angle instead of angle, frequency

# plot inter aural time difference alongside the angle in the legend
# comparing performance of different frequencies at different angles

# change color for each and put on one graph - DONE
# log distribution of 25 points from 0 to 8000 Hz
# real world data try out
# cutoff by counting
for i in range(5):
    plt.scatter(metadata[1], angle_mean[i], color=metadata[5][i], label=f"angle = {metadata[0][i]}")
    plt.axhline(angle_theory[i], color=metadata[5][i])

    plt.title(f"Performance of zero crossings method for angles")
    plt.xlabel("Frequency (Hz)")
    plt.xlim([0, 1000])
    plt.ylabel("Angle")

plt.legend()
plt.show()

# create graph of interaural level difference as a function of angle, by frequency
# polar plot to compare

for i in range(6):
    plt.scatter(metadata[0], level_mean[i], color=metadata[5][i], label=f"frequency = {metadata[1][i]}")
    plt.plot(metadata[0], level_mean[i], color=metadata[5][i])

    plt.title(f"Evolution of maximum ratio of leftchannel over rightchannel by frequency")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Left Intensity/Right Intensity")

plt.legend()
plt.show()

# filter by frequencies
