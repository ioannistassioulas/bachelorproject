import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import rc

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
plt.axhline(sample[-2], label= coordinates + f" = {angles[-2], sample[-2]}", color="red")
plt.xlabel("Direction of Arrival " + r'$\theta$ [$^{\circ}$]')
plt.ylabel("Sample rate " + r'$\Delta t^{-1} [Hz]$')
plt.title("Minimal sample rate for time difference encoding")
plt.legend()
plt.show()
