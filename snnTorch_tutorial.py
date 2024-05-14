import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np


# practice encoding spikes from visual dataset



volt = 0.9
volt_trace = []

for i in range(100):
    volt_trace.append(volt)
    volt = forward_euler_lif(volt)

# plt.plot(np.arange(100), volt_trace)
# plt.xlabel("Membrane Potential")
# plt.ylabel("Timestep")
# plt.title("Leaky Neuron Model")
# plt.show()

lif1 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-4, threshold=0.5)

# pulse input

current = torch.cat((torch.zeros(10, 1), torch.ones(190, 1) * 0.5), 0)
mem_volt = torch.zeros(1)
mem_rec = []
spk_rec = []

for step in range(len(current)):
    spikes, mem_volt = lif1(current[step], mem_volt)
    mem_rec.append(mem_volt)
    spk_rec.append(spikes)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec) * 0.5

plt.plot(np.arange(len(current)), mem_rec)
plt.plot(np.arange(len(current)), current.numpy())
plt.scatter(np.arange(len(current)), spk_rec)
plt.ylabel("current / voltage / spikenumber")
plt.xlabel("timestep")
plt.title("Leaky Neuron Model")
plt.show()

# start creating TDE model

def forward_euler_lif(v, resist=5e7, cap=1e-10, current=0, time_step=1e-3):
    time_constant = resist * cap
    v = v + time_step * (current * resist - v) / time_constant
    return v


def tde_neuron():


def tde_model(time, facilitatory, trigger, sr):
    """
    :param time: how long the recording for spike input takes place
    :param sr: sampling rate of the audio, taken via librosa
    :param facilitatory: spike train of first input that is received
    :param trigger: spike train of second input that is received
    :return:
    """

    # create LiF neuron for the TDE
    tde_neuron =

    # define timesteps where model will be defined over
    timesteps = sr * time

    # create array of timesteps to send spikes and create spike train for facilitatory input
    facilitatory = facilitatory * sr
    facilitatory = facilitatory.astype(int)

    current_f = torch.zeros(timesteps)
    for j in facilitatory:
        current_f[j] = 1  # at each index of the time step you input in facilitatory array

    # vice versa for trigger input
    trigger = trigger * sr
    trigger = trigger.astype(int)

    current_t = torch.zeros(timesteps)
    for j in trigger:
        current_t[j] = 1

    # create recording bank for spikes and membrane potential. Also, V(0) = 0
    spike_rec = []
    membrane_rec = []
    membrane_volt = torch.zeros(1)

    # Generate spikes for LiF model
    for steps in range(timesteps):
        spike, membrane_volt = tde_neuron(R=5.1, C=5e-3, time_step=1e-4, threshold=1)
        spike_rec.append(spike)
        membrane_rec.append(membrane_volt)

    #
