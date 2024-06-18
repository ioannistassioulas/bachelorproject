import audio_processing
from audio_processing import *
import time as t


# start recording time
start_time = t.time()

# create sound waves

frequency = np.arange(500, 12501, 1000)
angles = np.arange(0, 91, 5)
sr = 41000
distance = 0.048
time = 0.1

# create arrays to store ITD and ILD information
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

        timer  = int(time * sr)
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
        tau_mem = torch.tensor(1)
        mem, spk, fac, trg = tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(timer), current_f, current_t)
        spike_rec.append(torch.stack((mem[0], spk[0], fac[0], trg[0])))

        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(np.arange(timer), fac[0], label="facilitatory")
        # ax[0].plot(np.arange(timer), trg[0], label="trigger")
        # ax[0].legend()
        # ax[0].set_title("TDE recording portion")
        #
        # ax[1].plot(np.arange(timer), mem[0], label="membrane")
        # ax[1].plot(np.arange(timer), spk[0], label="spike")
        # ax[1].legend()
        # ax[1].set_title("Membrane and Spiking behaviour")
        #
        # fig.suptitle(f"Spiking behaviour for frequency = {frequency[j]}, angle = {angles[i]}")
        # plt.show()

        # check to make sure of progress are still working
        gc.collect()
        print(f"{k/(len(angles) * len(frequency))*100}% Time elapse = {t.time() - start_time}s \nAngle = {angles[i]} \nFrequency = {frequency[j]}")
        k += 1

    itd_real.append(np.mean(itd_mem))
    ild_real.append(np.mean(ild_mem))

    spike_rec = torch.stack(spike_rec)
    spike_mem.append(spike_rec)

spike_mem = torch.stack(spike_mem)
spike_mem = spike_mem.permute((1, 0, 2, 3))
print(f"Finished encoding! Time elapsed:{t.time() - start_time}s")

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

        total_spike = torch.tensor(signal.convolve(spk, mem))
        total_spike_count = torch.sum(total_spike)
        spike_result.append(total_spike_count)

        z += 1

    plt.plot(itd_real, spike_result, label=f"Frequency = {frequency[y]}")
    plt.scatter(itd_real, spike_result)
    y += 1

plt.xlabel("$\Delta t$")
plt.ylabel("Spike #")
plt.title("TDE performance rate")
plt.legend()
plt.show()

# try again
