# Figure out how to use Torch properly to create your own neural network
import snntorch.spikegen

import audio_processing
from audio_processing import *

import snntorch as snn
import torch

stereo, sampling_rate = librosa.load("/home/ioannistassioulas/Downloads/PDFs for Thesis/Thesis Audio/first_sample.wav", mono=False)
audio_processing.display_waves(stereo, sampling_rate)
left_mic, right_mic = audio_processing.count_peak_diff(stereo)  # create tensor of spike info

left_tensor = torch.Tensor(left_mic)
right_tensor = torch.Tensor(right_mic)

print(left_tensor)
# left_spikes = snntorch.spikegen.rate(left_tensor, time_var_input=True)  # creates spike train
# right_spikes = snntorch.spikegen.rate(right_tensor, time_var_input=True)
#
# print(left_tensor)
# print(left_spikes)
