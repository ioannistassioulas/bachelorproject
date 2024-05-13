
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

# practice encoding spikes from visual dataset


def forward_euler_lif(v, resist=5e7, cap=1e-10, current=0, time_step=1e-3):
    time_constant = resist * cap
    v = v + time_step * (current * resist - v) / time_constant
    return v


volt = 0.9
volt_trace = []

for i in range(100):
    volt_trace.append(volt)
    volt = forward_euler_lif(volt)

plt.plot(np.arange(100), volt_trace)
plt.xlabel("Membrane Potential")
plt.ylabel("Timestep")
plt.title("Leaky Neuron Model")
plt.show()

lif1 = snn.Lapicque(R=5, C=1e-3, time_step=1e-3)

mem_volt = torch.ones(1)*0.9
current = torch.zeros(100, 1)
spikes = torch.zeros(1)

mem_rec = [mem_volt]
for step in range(100):
    spikes, mem_volt = lif1(current[step], mem_volt)
    mem_rec.append(mem_volt)

mem_rec = torch.stack(mem_rec)
plt.plot(np.arange(101), mem_rec)
plt.xlabel("Membrane Potential")
plt.ylabel("Timestep")
plt.title("Leaky Neuron Model")
plt.show()
