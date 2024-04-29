# Figure out how to use Torch properly to create your own neural network
import snntorch.spikegen

import audio_processing
from audio_processing import *

import snntorch as snn
import torch

stereo, sampling_rate = librosa.load("/home/s4290755/Downloads/sample_sound_data.wav", mono=False)
channel_spikes = torch.Tensor(audio_processing.count_peak(stereo))  # create tensor of spike info
snntorch.spikegen.rate(channel_spikes, )

