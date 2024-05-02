# Figure out how to use Torch properly to create your own neural network
import snntorch.spikegen

import audio_processing
from audio_processing import *
import snntorch as snn
import torch

stereo, sampling_rate = librosa.load("/home/ioannistassioulas/PycharmProjects/pythonProject/bachelorproject/datasets"
                                     "/30_deg_200_Hz.wav", mono=False)
spike_data, spike_values = audio_processing.zero_crossing(stereo, sampling_rate)  # generate spike data values
time_difference = np.abs(spike_data[1][1:] - spike_data[0])
ground_truth = np.cos(np.pi/6) * 0.3 / 343

plt.plot(spike_data[0], time_difference)  # measured timestep difference
plt.axhline(ground_truth, color="red")  # ground truth as per our formula
plt.show()

# audio_processing.display_waves(stereo, sampling_rate)
# plt.figure().set_figwidth(12)
# librosa.display.waveshow(stereo[0], sr=sampling_rate, marker=".", color="blue", axis="time")
# plt.scatter(timestep, [0] * len(timestep), label="points", color="red")
# plt.xlim([0, 0.1])
# plt.legend()
# plt.show()

# left_tensor = torch.Tensor(left_mic)
# right_tensor = torch.Tensor(right_mic)

