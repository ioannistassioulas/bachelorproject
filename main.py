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

plt.scatter(angles, sample, color="black")
plt.plot(angles, sample, label="Required sample rate", color="black")
plt.axhline(48000, label="Native sample rate", color="blue")
coordinates = r'($\theta$, $\Delta t^{-1}$)'
plt.xlabel("Direction of Arrival " + r'$\theta$ [$^{\circ}$]')
plt.ylabel("Sample rate " + r'$\Delta t^{-1} [Hz]$')
plt.title("Minimal sample rate for time difference encoding")
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

