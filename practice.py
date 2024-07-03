from scipy import fft
import numpy as np
import matplotlib.pyplot as plt

# # how to filter found sound wave
# n = 5000
# x = np.linspace(0, 5, n)
# y = np.sin(2 * np.pi * 10 * x) + np.sin(2 * np.pi * 5 * x)
#
# plt.plot(x, y)
# plt.plot(x, np.sin(2 * np.pi * 5 * x))
# plt.show()
#
# xf = fft.fftfreq(y.size, d=x[1] - x[0])
# yf = fft.rfft(y)[: n // 2]
# yf[xf[: n // 2] > 5] = 0
# yfil = fft.irfft(yf)
# plt.plot(x[:len(yfil)], yfil)
#
# plt.plot(x, np.sin(2* np.pi * 5 * x))
# plt.show()

dictionary = {
    "color" : [10],
    "brand" : [20],
    "target audience": [30]
}
print(dictionary)
dictionary["color"].append("blue")
print(dictionary)
