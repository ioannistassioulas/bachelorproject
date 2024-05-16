import os
import audio_processing as ap

import audio_processing
from audio_processing import *

metadata = np.array([[15, 30, 45, 60, 75], [200, 500, 800, 2000, 5000, 8000],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372]],
                    dtype=object)

# create sound waves
sr = 44100


# Assume distance is approximately 1 meter away from origin
angles = np.linspace(-180, 180, 37)
frequency = np.linspace(0, 10000, 51)
t = np.linspace(0, 5, 5*sr)
ild_test = [None] * 36

waveform_1 = np.zeros([len(angles)-1, len(frequency)-1], dtype=object)
waveform_2 = np.zeros([len(angles)-1, len(frequency)-1], dtype=object)
# cwr = os.getcwd()
# synth = cwr + "/datasets/synthetic_data/"
# waves = [[None] * 6] * 5
# for i in range(len(metadata[0, power remains constant, while area changes. In case of the point source, area increases as A∝r2, while in case of the linear source, area increases as A∝ρ!])):
#     for j in range(len(metadata[1])):
#         file = audio_processing.name_parse(synth, metadata[0][i], metadata[1][j])
#         waves[i][j], sr = librosa.load(file, mono=False)

#inverse square law checking for difference
k = 0
for i in range(len(angles)-1):  # loop through angles
    theta = angles[i]
    radius = np.array([np.cos(theta), np.sin(theta), 0])
    left = radius + np.array([0.15, 0, 0])
    right = radius + np.array([-0.15, 0, 0])

    phase = np.cos(theta) * 0.3  /343
    amp_left = 1 / left.dot(left)
    amp_right = 1 / right.dot(right)

    ild_test[i] = np.abs(amp_left - amp_right)

    for j in range(len(frequency) - 1):  # loop through frequencies
        waveform_1[i][j] = amp_right * np.sin(t * frequency[j + 1])  # left channel
        waveform_2[i][j] = amp_left * np.sin((t - phase) * frequency[j + 1])  # right channel
        print(f"{k/1800*100}%")
        k += 1



# create arrays to store ITD and ILD information
time_difference = np.zeros([len(angles)-1, (len(frequency) - 1)])
level_difference = np.zeros([len(angles)-1, (len(frequency) - 1)])
count = 1
# go through all angles and frequencies again and apply tests
for i in range(len(angles)-1):  # by angle
    for j in range(len(frequency)-1):  # by frequency

        cutoff = 0
        endtime = 99
        waves = np.array([waveform_1[i][j], waveform_2[i][j]])
        spike_data, spike_values = audio_processing.zero_crossing(waves, sr)
        spike_x, spike_y = audio_processing.peak_difference(waves, sr)

        # fix any broadcasting issues
        spike_data, spike_values = audio_processing.fix_broadcasting(spike_data, spike_values)
        spike_x, spike_y = audio_processing.fix_broadcasting(spike_x, spike_y)

        # implement cutoff point to prevent extraneous answers
        spike_data, spike_values = audio_processing.set_recording(spike_data, spike_values, cutoff, endtime)
        spike_x, spike_y = audio_processing.set_recording(spike_x, spike_y, cutoff, endtime)

        # determine the inter aural time difference from the data amassed
        time_differences = spike_data[1] - spike_data[0]  # find difference in zero crossings from both channels
        time_difference[i][j] = np.mean(time_differences)  # return mean into metadata super array

        level_differences = np.abs(spike_y[1] - spike_y[0])
        level_difference[i][j] = np.mean(level_differences)

        #check to make surehings are still working
        print(f"{count/1800*100}%")
        count += 1

angle_avg_itd = np.rad2deg(audio_processing.angle_itd(0.3, time_difference))
angle_avg_ild = np.rad2deg(audio_processing.angle_ild(0.3, level_difference))

# generate graphs for ITD

for m in range(len(angles)-1):
    # stops working at 3840 Hz, 4160 Hz, 5120 Hz, 7360 Hz
    # Corresponds to wavelengths of 0.0002604 m, 0.00024038, 0.0001953125, 0.000135869
    plt.scatter(frequency[1:], angle_avg_itd[m], label=f"angle = {angles[m]}")
    plt.axhline(angles[m])

    plt.title(f"Performance of zero crossings method for angles, cutoff = {cutoff}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle")

plt.legend(loc='upper left')
plt.show()

# likewise for ILD

for n in range(len(angles)-1):
    plt.scatter(frequency[1:], level_difference[n], label=f"Angles = {angles[n]}")
    plt.axhline(ild_test[n])
    plt.title(f"Difference in peak per frequency level")
    plt.xlabel("Frequencies")
    plt.ylabel("ILD")

plt.legend(loc='upper right')
plt.show()

