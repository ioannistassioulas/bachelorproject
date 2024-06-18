import audio_processing
from audio_processing import *
import time as t

import gc  # garbage collection
# start recording time
start_time = t.time()

# create sound waves

frequency = np.linspace(250, 5000, 20)
angles = np.linspace(0, 90, 10)
sr = 7000
distance = 0.3
time = 0.1

# create arrays to store ITD and ILD information
# time_difference = np.zeros([len(angles), len(frequency)])
# level_difference = np.zeros([len(angles), len(frequency)])
spike_mem = []
itd_real = []
ild_real = []

# see above about counter
k = 1

# go through all angles and frequencies again and apply tests
for i in range(len(angles)):  # by angle

    itd_mem = []
    ild_mem = []
    spike_rec = []

    for j in range(len(frequency)):  # by frequency
        l, r, itd, ild = audio_processing.generate_test_waves(angles[i], frequency[j], sr, time, distance)
        itd_mem.append(itd)
        ild_mem.append(ild)

        waves = np.array([l, r])

        spike_data, spike_values = audio_processing.zero_crossing(waves, sr)
        spike_x, spike_y = audio_processing.peak_difference(waves, sr)

        # fix any broadcasting issues
        spike_data, spike_values = audio_processing.fix_broadcasting(spike_data, spike_values)
        spike_x, spike_y = audio_processing.fix_broadcasting(spike_x, spike_y)

        # # determine the inter aural time difference from the data amassed
        # time_differences = spike_data[1] - spike_data[0]  # find difference in zero crossings from both channels
        # time_difference[i][j] = np.mean(time_differences)  # return mean into metadata super array
        #
        # level_differences = spike_y[1] - spike_y[0]
        # level_difference[i][j] = np.mean(level_differences)
        # print("done!")
        # start making spikes
        # transfer zeros into spike train
        spike_data[0] = (spike_data[0]) * sr  # subtract by start time to keep length consistent with zeros
        current_f = torch.zeros(time * sr)
        for n in spike_data[0]:
            current_f[int(n)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        # repeating for trigger input
        spike_data[1] = (spike_data[1]) * sr
        current_t = torch.zeros(time * sr)
        for n in spike_data[1]:
            current_t[int(n)] = torch.ones(1)

        # pass everything into tde
        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.005)
        mem, spk, fac, trg = tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(time * sr), current_f, current_t)
        spike_rec.append([mem[0], spk[0], fac[0], trg[0]])

        # check to make sure of progress are still working
        gc.collect()
        print(f"{k/(len(angles) * len(frequency))*100}% Time elapse = {t.time() - start_time}s \nAngle = {angles[i]} \nFrequency = {frequency[j]}")
        k += 1

    itd_real.append(np.mean(itd_mem))
    ild_real.append(np.mean(ild_mem))
    spike_mem.append(spike_rec)

print(f"Finsihed encoding! Time elapsed:{t.time() - start_time}s")

# at the moment, the array is in angle * frequency, so transpose
spike_mem = spike_mem.transpose()
# go through each frequency and count the total number of spikes
for i in spike_mem:  # per frequency
    spike_result = []
    for j in i:  # per angle
        mem = j[0]
        spk = j[1]
        fac = j[2]
        trg = j[3]

        total_spike_count = torch.sum(spk)
        spike_result.append(torch.sum)

    plt.plot(angles, spike_result)
    plt.xlabel("Angle")
    plt.ylabel("Spike #")
    plt.title("Spiking activity of TDE")
print(np.array(spike_mem).shape)

