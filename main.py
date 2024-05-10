import os
import audio_processing as ap

import audio_processing
from audio_processing import *

metadata = np.array([[15, 30, 45, 60, 75], [200, 500, 800, 2000, 5000, 8000],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372]],
                    dtype=object)

# create sound waves
sr = 44100
waveform_1 = np.zeros([5, 25], dtype=object)
waveform_2 = np.zeros([5, 25], dtype=object)

frequency = np.linspace(0, 8000, 26)
t = np.linspace(0, 1, sr)

# cwr = os.getcwd()
# synth = cwr + "/datasets/synthetic_data/"
# waves = [[None] * 6] * 5
# for i in range(len(metadata[0])):
#     for j in range(len(metadata[1])):
#         file = audio_processing.name_parse(synth, metadata[0][i], metadata[1][j])
#         waves[i][j], sr = librosa.load(file, mono=False)


for i in range(len(metadata[0])):  # loop through angles
    for j in range(len(frequency) - 1):  # loop through frequencies
        phase = metadata[2][i]
        waveform_1[i][j] = np.sin(t * frequency[j + 1])  # left channel
        waveform_2[i][j] = np.sin((t - phase) * frequency[j + 1])  # right channel


# create arrays to store ITD and ILD information
time_difference = np.zeros([len(metadata[0]), (len(frequency) - 1)])
angle_mean = np.zeros([len(metadata[0]), (len(frequency) - 1)])
level_difference = np.zeros([len(metadata[0]), (len(frequency) - 1)])


# go through all angles and frequencies again and apply tests
for i in range(len(metadata[0])):  # by angle
    for j in range(len(frequency)-1):  # by frequency

        cutoff = 0.01
        waves = np.array([waveform_1[i][j], waveform_2[i][j]])
        spike_data, spike_values = audio_processing.zero_crossing(waves, sr)

        # fix any broadcasting issues
        length = sorted([spike_data[0], spike_data[1]], key=len)
        spike_data[0] = spike_data[0][:len(length[0])]
        spike_data[1] = spike_data[1][:len(length[0])]
        spike_values[0] = spike_values[0][:len(length[0])]
        spike_values[1] = spike_values[1][:len(length[0])]

        # implement cutoff point to prevent extraneous answers
        booler = spike_data[0] > cutoff
        spike_data[0] = spike_data[0][booler]
        spike_data[1] = spike_data[1][booler]
        spike_values[0] = spike_values[0][booler]
        spike_values[1] = spike_values[1][booler]

        # plt.plot(t, waveform_1[i][j])
        # plt.plot(t, waveform_2[i][j])
        # plt.scatter(spike_data[1], spike_values[1])
        # plt.scatter(spike_data[0], spike_values[0])
        # plt.show()

        # determine the inter aural time difference from the data amassed
        time_differences = np.abs(spike_data[1] - spike_data[0])  # find difference in zero crossings from both channels
        time_difference[i][j] = np.mean(time_differences)  # return mean into metadata super array
        angle_mean[i][j] = np.rad2deg(audio_processing.angle_by_itd(0.3, time_difference[i][j]))

        # determine the inter aural level difference from data amassed per sample step
        ratios = audio_processing.count_peak_ratio(waves)
        # max_level_differences = np.mean(ratios)
        max_level_differences = audio_processing.peak_diff(waves)
        level_difference[i][j] = np.mean(max_level_differences)

# prepare colors to use for the graph
colors = list(mcolors.BASE_COLORS)

# generate graphs for ITD and ILD respectively
for m in range(5):
    plt.scatter(frequency[1:], angle_mean[m], color=colors[m], label=f"angle = {metadata[0][m]}")
    plt.axhline(metadata[0][m], color=colors[m])

    plt.title(f"Performance of zero crossings method for angles, cutoff = {cutoff}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle")

plt.legend(loc='upper left')
plt.show()

colors = list(mcolors.XKCD_COLORS)

for n in range(5):
    plt.scatter(frequency[1:], level_difference[n], color=colors[n+15], label=f"Angles = {metadata[0][n]}")
    plt.plot(frequency[1:], level_difference[n], color=colors[n+15])

    plt.title(f"Average difference in left channel and right channel audio intensity")
    plt.xlabel("Angles")
    plt.ylabel("Left Intensity - Right Intensity")

plt.legend(loc='upper right')
plt.show()
