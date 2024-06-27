# import audio_processing
# from audio_processing import *
import numpy as np
import audio_processing as ap
import torch
import tde_model as tde
import gc
from scipy import signal
import matplotlib.pyplot as plt
import time as t
from scipy import signal
from itertools import groupby


# start recording time
start_time = t.time()

# create sound waves

frequency = np.arange(500, 5501, 1000)
angles = np.arange(0, 81, 5)
sr = 42000
distance = 0.048
time = 0.1

# create arrays to store ITD and ILD information
spike_mem = []
itd_real = []
ild_real = []
time_difference = []
err = []
# see above about counter
k = 1

# go through all angles and frequencies again and apply tests
for i in range(len(angles)):  # by angle

    itd_mem = []
    ild_mem = []
    spike_rec = []
    tim_mem = []
    err_m = []
    for j in range(len(frequency)):  # by frequency
        l, r, itd, ild = ap.generate_test_waves(angles[i], frequency[j], sr, time, distance)
        itd_mem.append(itd)
        ild_mem.append(ild)

        waves = np.array([l, r])

        spike_data, spike_values = ap.zero_crossing(waves, sr)
        spike_x, spike_y = ap.peak_difference(waves, sr)

        # fix any broadcasting issues
        spike_data, spike_values = ap.fix_broadcasting(spike_data, spike_values)
        spike_x, spike_y = ap.fix_broadcasting(spike_x, spike_y)

        # determine the inter aural time difference from the data amassed
        time_differences = spike_data[1] - spike_data[0]  # find difference in zero crossings from both channels
        tim_mem.append(np.mean(time_differences))  # return mean into metadata super array
        err_m.append(np.std(time_differences))

        # transfer zeros into spike train
        timer = int(time * sr)

        spike_data[0] = (spike_data[0]) * sr  # subtract by start time to keep length consistent with zeros
        current_f = torch.zeros(timer)
        for n in spike_data[0]:
            current_f[int(n)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        # repeating for trigger input
        spike_data[1] = (spike_data[1]) * sr
        current_t = torch.zeros(timer)
        for n in spike_data[1]:
            current_t[int(n)] = torch.ones(1)

        # pass everything into tde
        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.01)
        mem, spk, fac, trg = tde.tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(timer), current_f, current_t)
        spike_rec.append(torch.stack((mem[0], spk[0], fac[0], trg[0])))

        # check to make sure of progress are still working
        gc.collect()
        print(f"{k/(len(angles) * len(frequency))*100}% Time elapse = {t.time() - start_time}s \nAngle = {angles[i]} \nFrequency = {frequency[j]}")
        k += 1

    itd_real.append(np.mean(itd_mem))
    ild_real.append(np.mean(ild_mem))
    time_difference.append(tim_mem)
    err.append(err_m)

    spike_rec = torch.stack(spike_rec)
    spike_mem.append(spike_rec)

spike_mem = torch.stack(spike_mem)
spike_mem = spike_mem.permute((1, 0, 2, 3))
print(f"Finished encoding! Time elapsed:{t.time() - start_time}s")

theta = np.cos(np.deg2rad(angles))
# make plot of itd
plt.scatter(angles, itd_real, label="Encoded ITD", color="black")
plt.plot(angles, distance / 343 * np.cos(np.deg2rad(angles)), label=r'$\frac{d}{v}\cos{(\theta)}$', color="red")
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$\theta$')
plt.title("ITD vs Angle")
plt.legend()
plt.show()

# go through each frequency and count the total number of spikes
y = 0

for i in spike_mem:  # per frequency
    spike_result = []
    z = 0
    for j in i:  # per angle
        mem = j[0]
        spk = j[1]
        fac = j[2]
        trg = j[3]
        tiempo = []

        # calculate isi of spikes
        for entry in range(len(spk)):
            if spk[entry]:
                tiempo.append(entry)
        isi = np.diff(tiempo)
        isifil = np.array([k for k, g in groupby(isi)])
        peaks = signal.argrelextrema(isifil, np.greater)

        diff = len(peaks[0])  # gaps btw groups therefore total groups = diff + 1

        print(f"for angle {angles[z]} with freq {frequency[y]}: {diff + 1}")

        # plt.plot(np.arange(len(isi)), isi)
        # plt.plot(np.arange(len(isifil)), isifil)
        # plt.show()
        spike_result.append(torch.sum(spk) / diff + 1)
        z += 1

    plt.plot(itd_real, spike_result, label=f"Frequency = {frequency[y]}")
    plt.scatter(itd_real, spike_result)
    plt.xlabel("$\Delta t$")
    plt.ylabel("# of Spikes")
    plt.title("TDE performance rate")
    plt.legend()
    plt.show()

    y += 1

# frequency 550, angle 0
i = 0
j = 0
fig, ax = plt.subplots(2)

ax[0].plot(np.arange(len(spike_mem[i][j][2])), spike_mem[i][j][2], label="facilitatory")
ax[0].plot(np.arange(len(spike_mem[i][j][3])), spike_mem[i][j][3], label="trigger")
ax[0].set_title("Facilitatory and Trigger Inputs")
ax[0].legend()

ax[1].plot(np.arange(len(spike_mem[i][j][0])), spike_mem[i][j][0], label="Membrane")
ax[1].plot(np.arange(len(spike_mem[i][j][1])), spike_mem[i][j][1], label="Spikes")
ax[1].axhline(y=0.7, color='blue', marker='_', label="threshold")
ax[1].set_title("LIF Neuron and corresponding spikes")
ax[1].legend()

fig.suptitle(f"TDE activity for Frequency of {frequency[i]}Hz and Angle of {angles[j]} degrees")

fig.text(0.5, 0.04, 'Time', ha='center')
fig.text(0.04, 0.5, 'Potential', va='center', rotation='vertical')

plt.show()

# frequency 550, angle 60
i = 0
j = 12
fig, ax = plt.subplots(2)

ax[0].plot(np.arange(len(spike_mem[i][j][2])), spike_mem[i][j][2], label="facilitatory")
ax[0].plot(np.arange(len(spike_mem[i][j][3])), spike_mem[i][j][3], label="trigger")
ax[0].set_title("Facilitatory and Trigger Inputs")
ax[0].legend()

ax[1].plot(np.arange(len(spike_mem[i][j][0])), spike_mem[i][j][0], label="Membrane")
ax[1].plot(np.arange(len(spike_mem[i][j][1])), spike_mem[i][j][1], label="Spikes")
ax[1].axhline(y=0.7, color='blue', marker='_', label="threshold")
ax[1].set_title("LIF Neuron and corresponding spikes")
ax[1].legend()

fig.suptitle(f"TDE activity for Frequency of {frequency[i]}Hz and Angle of {angles[j]} degrees")

fig.text(0.5, 0.04, 'Time', ha='center')
fig.text(0.04, 0.5, 'Potential', va='center', rotation='vertical')

plt.show()

# frequency 3500, angle 0
i = 3
j = 0
fig, ax = plt.subplots(2)

ax[0].plot(np.arange(len(spike_mem[i][j][2])), spike_mem[i][j][2], label="facilitatory")
ax[0].plot(np.arange(len(spike_mem[i][j][3])), spike_mem[i][j][3], label="trigger")
ax[0].set_title("Facilitatory and Trigger Inputs")
ax[0].legend()

ax[1].plot(np.arange(len(spike_mem[i][j][0])), spike_mem[i][j][0], label="Membrane")
ax[1].plot(np.arange(len(spike_mem[i][j][1])), spike_mem[i][j][1], label="Spikes")
ax[1].axhline(y=0.7, color='blue', marker='_', label="threshold")
ax[1].set_title("LIF Neuron and corresponding spikes")
ax[1].legend()


fig.suptitle(f"TDE activity for Frequency of {frequency[i]}Hz and Angle of {angles[j]} degrees")
fig.text(0.5, 0.04, 'Time', ha='center')
fig.text(0.04, 0.5, 'Potential', va='center', rotation='vertical')

plt.show()


plt.plot(angles, )
