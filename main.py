import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# final file to create all plots
# create plot showing sample rate needed to encode time difference
angles = np.arange(5, 90, 10)
v = 343
d = 0.048

sample = (v / d) * (1 / np.cos(np.deg2rad(angles)))
print(sample)

plt.scatter(angles, sample / 1000, color="black")
plt.plot(angles, sample / 1000, label="Required sample rate", color="black")
plt.axhline(48, label="Native sample rate", color="blue")
coordinates = r'($\theta$, $\Delta t^{-1}$)'
plt.xlabel("Direction of Arrival " + r'$\theta$ [$^{\circ}$]')
plt.ylabel("Sample rate " + r'$\Delta t^{-1} [kHz]$')
plt.title("Minimal sample rate for time difference encoding")

plt.grid()
plt.legend()
plt.show()


# # show effects of down sampling:
# home = os.getcwd() + "/results/"
# data = pd.read_csv(home + "Down-Sampling-1.csv")
# sig = [data['IBL'].to_numpy(), data['IAL'].to_numpy()]
#
# sr_before = int(0.01 * 48000)
# sr_after = int(0.01 * 48000/2)
#
# plt.plot(np.linspace(0, 0.01, sr_before), sig[0][:sr_before], label="Before downsampling")
# plt.plot(np.linspace(0, 0.01, sr_after), sig[1][:sr_after], label="After downsampling")
# plt.xlabel("Time")
# plt.ylabel("Intensity")
# plt.title("Effects of downsampling on sound wave")
# plt.plot()
#
# plt.legend()
# plt.show()
#
# home = home + '/results/'
# stereo = pd.read_csv(home + f"Down-Sampling-{i+1}.csv")
#
# sig = [stereo['IAL'].to_numpy(), stereo['IAR'].to_numpy()]
# sig = [sig[0][~np.isnan(sig[0])], sig[1][~np.isnan(sig[1])]]
#
# fig, ax = plt.subplots(2)
# ax[0].plot(np.arange(len(sig[0][int(start* sr) : int((start*sr + index))])), sig[0][int(start* sr) : int((start*sr + index))], label="left")
# ax[0].plot(np.arange(len(sig[1][int(start* sr) : int((start*sr + index))])), sig[1][int(start* sr) : int((start*sr + index))], label="right")
# ax[0].set_title("Unfiltered")
# ax[0].legend()
#
# ax[1].plot(np.arange(len(wave_fft[0])), wave_fft[0], color='red', label="left")
# ax[1].plot(np.arange(len(wave_fft[1])), wave_fft[1], color='blue', label="right")
# ax[1].scatter(zero_x[0], [0] * len(zero_x[0]), color='red')
# ax[1].scatter(zero_x[1], [0] * len(zero_x[1]), color='blue')
# ax[1].set_title("Filtered Sound Data")
# ax[1].legend()

# spikers = np.array(spikers).transpose()
# plt.scatter(spikers[1], spikers[0])
# plt.show()


# fig, ax = plt.subplots(4)
# ax[0].plot(np.arange(len(sig[0][int(start* sr) : int((start*sr + index))])), sig[0][int(start* sr) : int((start*sr + index))], label="left")
# ax[0].plot(np.arange(len(sig[1][int(start* sr) : int((start*sr + index))])), sig[1][int(start* sr) : int((start*sr + index))], label="right")
# ax[0].set_title("Unfiltered")
# ax[0].legend()
#
# ax[1].plot(np.arange(len(wave_fft[0])), wave_fft[0], color='red', label="left")
# ax[1].plot(np.arange(len(wave_fft[1])), wave_fft[1], color='blue', label="right")
# ax[1].scatter(zero_x[0], [0] * len(zero_x[0]), color='red')
# ax[1].scatter(zero_x[1], [0] * len(zero_x[1]), color='blue')
# ax[1].set_title("Filtered Sound Data")
# ax[1].legend()
#
# ax[2].plot(np.arange(len(fac[0])), fac[0], label="facilitatory")
# ax[2].plot(np.arange(len(trg[0])), trg[0], label="trigger")
# ax[2].set_title("Facilitatory and Trigger Inputs")
# ax[2].legend()
#
# ax[3].plot(np.arange(len(mem[0])), mem[0], label="Membrane")
# ax[3].plot(np.arange(len(spk[0])), spk[0], label="Spikes")
# ax[3].axhline(y=0.7, color='blue', marker='_', label="threshold")
# ax[3].set_title("LIF Neuron and corresponding spikes")
# ax[3].legend()
#
# fig.suptitle(f"TDE activity for Sound File {i+1}, Sound Event {j+1}; Angle = {45 - angle}. Average per group = {avg_spk_per_group}")
#
# fig.text(0.5, 0.04, 'Time', ha='center')
# fig.text(0.04, 0.5, 'Potential', va='center', rotation='vertical')
#
# plt.show()

# wave_fft = []
# for wave in waves:
#     wave_f = fft.rfft(wave)
#     freq_f = fft.fftfreq(len(wave), 1/sr)[:int(len(wave) // 2)]
#
#     wave_f = wave_f[:len(freq_f)]
#     wave_f[(freq_f > band_gap[1])] = 0
#     wave_f[(freq_f < band_gap[0])] = 0
#     new_wave = fft.irfft(wave_f)
#     wave_fft.append(new_wave)

# l_max = np.max(wave_fft[0])
# r_max = np.max(wave_fft[1])
# main_freq_l = np.abs(freq_fft[wave_fft[0].argmax()])  # main frequency
# main_freq_r = np.abs(freq_fft[wave_fft[1].argmax()])  # main frequency
# avg_freq = int(np.mean([main_freq_l, main_freq_r]))

# fig, ax = plt.subplots(2)
# ax[0].plot(np.arange(len(sig[0][int(start* sr) : int((start*sr + index))])), sig[0][int(start* sr) : int((start*sr + index))], label="Left")
# ax[0].plot(np.arange(len(sig[1][int(start* sr) : int((start*sr + index))])), sig[1][int(start* sr) : int((start*sr + index))], label="Right")
# ax[0].set_title("Unfiltered")
# ax[0].legend()
# ax[0].grid()
#
# ax[1].plot(np.arange(len(wave_fft[0])), wave_fft[0], color='red', label="Left")
# ax[1].plot(np.arange(len(wave_fft[1])), wave_fft[1], color='blue', label="Right")
# ax[1].scatter(zero_x[0], [0] * len(zero_x[0]), color='red')
# ax[1].scatter(zero_x[1], [0] * len(zero_x[1]), color='blue')
# ax[1].set_title("Filtered")
# ax[1].legend()
# ax[1].grid()
#
# fig.text(0.5, 0.04, 'Time', ha='center')
# fig.text(0.04, 0.5, 'Sound Intensity', va='center', rotation='vertical')
#
# plt.show()
