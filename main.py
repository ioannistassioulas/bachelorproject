# #
# from brian2 import *
#
# start_scope()
#
# N = 10
# # eqs = '''
# # dv/dt = (current-v)/tau : 1
# # current : 1
# # tau : second
# # '''
#
# G = NeuronGroup(N, 'v:1')
# # G.current = [2, 0, 0]
# # G.tau = [10, 100, 100] * ms
#
# synapse = Synapses(G, G)
# synapse.connect(condition='abs(i-j)<4 and i!=j')
#
# synapse1 = Synapses(G, G)
# synapse1.connect(j='i')
# M = StateMonitor(G, 'v', record=True)
#
# run(100 * ms)
#
#
# def visualize_connections(s):
#     number_source = len(s.source)
#     number_target = len(s.target)
#
#     figure(figsize=(10, 4))
#     subplot(121)
#
#     plot(zeros(number_source), arange(number_source), 'ok', ms=10)  # put points of sources on the map
#     plot(ones(number_target), arange(number_target), 'ok', ms=10)  # put points of targets on the map
#
#     for i, j in zip(s.i, s.j):
#         plot([0, 1], [i, j],
#              '-k')  # for each connection created by the synapse, plot a line from the source(0, source) to target
#         # (1, target)
#
#     xticks([0, 1], ['Source', 'Target'])  # label of sources and targets
#     ylabel('Neuron index')  #
#     xlim(-0.1, 1.1)
#     ylim(-1, max(number_source, number_target))
#
#     subplot(222)
#     plot(s.i, s.j, 'ok')
#     xlim(-1, number_source)
#     ylim(-1, number_target)
#     xlabel('Source neurons')
#     ylabel('Target neurons')
#
#
# visualize_connections(synapse)
# visualize_connections(synapse1)
# show()
#
# # plot(M.t/ms, M.v[0], label="Neuron 0")
# # plot(M.t/ms, M.v[1], label="Neuron 1")
# # plot(M.t/ms, M.v[2], label="Neuron 2")
# # xlabel("Time [ms]")
# # ylabel("v")
# # legend()
#
# # show()

import os
import audio_processing as ap

import audio_processing
from audio_processing import *

metadata = np.array([[15, 30, 45, 60, 75], [200, 500, 800, 2000, 5000, 8000],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372],
                     np.zeros(5 * 6), np.zeros(5 * 6), ["red", "blue", "orange", "green", "black", "brown"]],
                    dtype=object)

# create sound waves
sr = 44100
waveform_1 = np.zeros([5, 25], dtype=object)
waveform_2 = np.zeros([5, 25], dtype=object)
waves = np.zeros(5, dtype=object)

frequency = np.linspace(0, 8000, 26)
t = np.linspace(0, 1, sr)

for i in range(len(metadata[0])):  # loop through angles
    for j in range(len(frequency) - 1):  # loop through frequencies
        phase = metadata[2][i]
        waveform_1[i][j] = np.sin(t * frequency[j + 1])  # left channel
        waveform_2[i][j] = np.sin((t - phase) * frequency[j + 1])  # right channel

time_difference = np.zeros(len(metadata[0]) * (len(frequency) - 1))
level_difference = np.zeros(len(metadata[0]) * (len(frequency) - 1))
for k in range(len(metadata[0])):
    for l in range(len(frequency) - 1):
        spike_data, spike_values = audio_processing.zero_crossing(np.array([waveform_1[k][l], waveform_2[k][l]]), sr,
                                                                  0.001)

        # fix any broadcasting issues
        length = sorted([spike_data[0], spike_data[1]], key=len)
        spike_data[0] = spike_data[0][:len(length[0])]
        spike_data[1] = spike_data[1][:len(length[0])]

        # determine the inter aural time difference from the data amassed
        time_differences = np.abs(spike_data[1] - spike_data[0])  # find difference in zero crossings from both channels
        time_difference[(k * 25) + l] = np.mean(time_differences)  # return mean into metadata super array

        # determine the inter aural level difference from data amassed per sample step
        max_level_differences = np.abs(
            np.nanmean(audio_processing.count_peak_ratio(np.array([waveform_1[k][l], waveform_2[k][l]]))))
        level_difference[(k * 25) + l] = max_level_differences

angle_mean = np.reshape(
    np.rad2deg(audio_processing.angle_by_itd(*time_difference)), [len(metadata[0]), len(frequency)-1])

angle_theory = np.rad2deg(audio_processing.angle_by_itd(*metadata[2]))

level_mean = np.reshape(level_difference, [len(metadata[0]), len(
    frequency) - 1]).transpose()  # now it goes frequency, angle instead of angle, frequency

for m in range(5):
    plt.scatter(frequency[1:], angle_mean[m], color=metadata[5][m], label=f"angle = {metadata[0][m]}")
    plt.axhline(angle_theory[m], color=metadata[5][m])

    plt.title(f"Performance of zero crossings method for angles")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle")

plt.legend()
plt.show()

for n in range(25):
    plt.scatter(metadata[0], level_mean[n], label=f"frequency = {frequency[n]}")
    plt.plot(metadata[0], level_mean[n])

    plt.title(f"Evolution of maximum ratio of leftchannel over rightchannel by frequency")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Left Intensity/Right Intensity")

plt.legend()
plt.show()
