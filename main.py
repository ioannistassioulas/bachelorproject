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
ild_test = [None] * 5
# Assume distance is approximately 1 meter away from origin
frequency = np.linspace(0, 20000, 26)
t = np.linspace(0, 5, 5*sr)

# cwr = os.getcwd()
# synth = cwr + "/datasets/synthetic_data/"
# waves = [[None] * 6] * 5
# for i in range(len(metadata[0, power remains constant, while area changes. In case of the point source, area increases as A∝r2, while in case of the linear source, area increases as A∝ρ!])):
#     for j in range(len(metadata[1])):
#         file = audio_processing.name_parse(synth, metadata[0][i], metadata[1][j])
#         waves[i][j], sr = librosa.load(file, mono=False)

#inverse square law checking for difference

for i in range(len(metadata[0])):  # loop through angles
    theta = metadata[0][i]
    radius = np.array([np.cos(theta), np.sin(theta), 0])
    left = radius + np.array([0.15, 0, 0])
    right = radius + np.array([-0.15, 0, 0])

    phase = metadata[2][i]
    amp_left = 1 / left.dot(left)
    amp_right = 1 / right.dot(right)

    ild_test[i] = np.abs(amp_left - amp_right)

    for j in range(len(frequency) - 1):  # loop through frequencies
        waveform_1[i][j] = amp_right * np.sin(t * frequency[j + 1])  # left channel
        waveform_2[i][j] = amp_left * np.sin((t - phase) * frequency[j + 1])  # right channel



# create arrays to store ITD and ILD information
time_difference = np.zeros([len(metadata[0]), (len(frequency) - 1)])
level_difference = np.zeros([len(metadata[0]), (len(frequency) - 1)])

# go through all angles and frequencies again and apply tests
for i in range(len(metadata[0])):  # by angle
    for j in range(len(frequency)-1):  # by frequency

        cutoff = 0
        endtime = 99
        waves = np.array([waveform_1[i][j], waveform_2[i][j]])
        spike_data, spike_values = audio_processing.zero_crossing(waves, sr)
        spike_x, spike_y = audio_processing.peak_difference(waves, sr)

        # fix any broadcasting issues
        length = sorted([spike_data[0], spike_data[1]], key=len)
        spike_data[0] = spike_data[0][:len(length[0])]
        spike_data[1] = spike_data[1][:len(length[0])]
        spike_values[0] = spike_values[0][:len(length[0])]
        spike_values[1] = spike_values[1][:len(length[0])]

        length = sorted([spike_x[0], spike_x[1]], key=len)
        spike_x[0] = spike_x[0][:len(length[0])]
        spike_x[1] = spike_x[1][:len(length[0])]
        spike_y[0] = spike_y[0][:len(length[0])]
        spike_y[1] = spike_y[1][:len(length[0])]

        # implement cutoff point to prevent extraneous answers
        start = spike_data[0] > cutoff
        stop = spike_data[0] < endtime
        spike_values[0] = spike_values[0][np.logical_and(start, stop)]
        spike_values[1] = spike_values[1][np.logical_and(start, stop)]
        spike_data[0] = spike_data[0][np.logical_and(start, stop)]
        spike_data[1] = spike_data[1][np.logical_and(start, stop)]

        start = spike_x[0] > cutoff
        stop = spike_x[0] < endtime
        spike_y[0] = spike_y[0][np.logical_and(start, stop)]
        spike_y[1] = spike_y[1][np.logical_and(start, stop)]
        spike_x[0] = spike_x[0][np.logical_and(start, stop)]
        spike_x[1] = spike_x[1][np.logical_and(start, stop)]

        plt.plot(t, waveform_1[i][j])
        plt.scatter(spike_x[0], spike_y[0])
        plt.plot(t, waveform_2[i][j])
        plt.scatter(spike_x[1], spike_y[1])
        plt.title(f"freq = {frequency[j+1]}, angle={metadata[0][i]}")
        plt.show()

        # determine the inter aural time difference from the data amassed
        time_differences = spike_data[1] - spike_data[0]  # find difference in zero crossings from both channels
        time_difference[i][j] = np.mean(time_differences)  # return mean into metadata super array

        level_differences = np.abs(spike_y[1] - spike_y[0])
        level_difference[i][j] = np.mean(level_differences)

print(level_difference)
angle_avg_itd = np.rad2deg(audio_processing.angle_itd(0.3, time_difference, 343))
angle_avg_ild = np.rad2deg(audio_processing.angle_ild(0.3, level_difference))

# generate graphs for ITD
colors = list(mcolors.BASE_COLORS)
for m in range(5):
    plt.scatter(frequency[1:], angle_avg_itd[m], color=colors[m], label=f"angle = {metadata[0][m]}")
    plt.axhline(metadata[0][m], color=colors[m])

    plt.title(f"Performance of zero crossings method for angles, cutoff = {cutoff}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle")

plt.legend(loc='upper left')
plt.show()

# likewise for ILD
colors = list(mcolors.XKCD_COLORS)
for n in range(5):
    plt.scatter(frequency[1:], level_difference[n], color=colors[n+15], label=f"Angles = {metadata[0][n]}")
    plt.axhline(ild_test[n], color=colors[n+15])
    plt.title(f"Difference in peak per frequency level")
    plt.xlabel("Frequencies")
    plt.ylabel("ILD")

plt.legend(loc='upper right')
plt.show()
