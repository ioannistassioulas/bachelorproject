import audio_processing
from audio_processing import *
import time as t
# start recording time
start_time = t.time()

# create sound waves

frequency = np.array([500, 2000, 5000])
angles = np.array([0, 30, 60, 90])
sr = 5000
distance = 0.3
time = 1

# create arrays to store ITD and ILD information
time_difference = np.zeros([len(angles), len(frequency)])
level_difference = np.zeros([len(angles), len(frequency)])
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
        cutoff = 0
        endtime = 99

        l, r, itd, ild = audio_processing.generate_test_waves(angles[i], frequency[j], sr, time, distance)
        itd_mem.append(itd)
        ild_mem.append(ild)

        waves = np.array([l, r])

        spike_data, spike_values = audio_processing.zero_crossing(waves, sr)
        spike_x, spike_y = audio_processing.peak_difference(waves, sr)

        # fix any broadcasting issues
        spike_data, spike_values = audio_processing.fix_broadcasting(spike_data, spike_values)
        spike_x, spike_y = audio_processing.fix_broadcasting(spike_x, spike_y)

        # implement cutoff point to prevent extraneous answers
        spike_data, spike_values = audio_processing.set_recording(spike_data, spike_values, cutoff, endtime)
        spike_x, spike_y = audio_processing.set_recording(spike_x, spike_y, cutoff, endtime)

        # determine the inter aural time difference from the data amassed
        time_differences = spike_data[1] - spike_data[0]  # find difference in zero crossings from both channels
        time_difference[i][j] = np.mean(time_differences)  # return mean into metadata super array

        level_differences = spike_y[1] - spike_y[0]
        level_difference[i][j] = np.mean(level_differences)
        print("done!")
        # start making spikes
        # transfer zeros into spike train
        spike_data[0] = (spike_data[0] - cutoff) * sr  # subtract by start time to keep length consistent with zeros
        current_f = torch.zeros(time * sr)
        for l in spike_data[0]:
            current_f[int(l)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        # repeating for trigger input
        spike_data[1] = (spike_data[1] - cutoff) * sr
        current_t = torch.zeros(time * sr)
        for l in spike_data[1]:
            current_t[int(l)] = torch.ones(1)

        # pass everything into tde
        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.005)
        mem, spk, fac, trg = tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(time * sr), current_f, current_t)
        spike_rec.append([mem[0], spk[0], fac[0], trg[0]])

        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(np.arange(time*sr), fac[0], label="facilitatory")
        # ax[0].plot(np.arange(time*sr), trg[0], label="trigger")
        # ax[0].legend()
        # ax[0].set_title("TDE recording portion")
        #
        # ax[1].plot(np.arange(time*sr), mem[0], label="membrane")
        # ax[1].plot(np.arange(time*sr), spk[0], label="spike")
        # ax[1].legend()
        # ax[1].set_title("Membrane and Spiking behaviour")
        #
        # fig.suptitle(f"Spiking behaviour for SIM: frequency={frequency[j]}, angle={angles[i]}")
        # plt.show()

        # check to make sure of progress are still working
        print(f"{k/(len(angles) * len(frequency))*100}% Time elapse = {t.time() - start_time}s")
        k += 1
    itd_real.append(np.mean(itd_mem))
    ild_real.append(np.mean(ild_mem))
    spike_mem.append(spike_rec)
print(f"Finsihed encoding! Time elapsed:{t.time() - start_time}s")

# compare tde by frequency
data = spike_mem[0][0]
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(time*sr), data[2], label="facilitatory")
ax[0].plot(np.arange(time*sr), data[3], label="trigger")
ax[0].legend()
ax[0].set_title("TDE recording portion")

ax[1].plot(np.arange(time*sr), data[0], label="membrane")
ax[1].plot(np.arange(time*sr), data[1], label="spike")
ax[1].legend()
ax[1].set_title("Membrane and Spiking behaviour")

fig.suptitle(f"Spiking behaviour for SIM: frequency={frequency[0]}, angle={angles[0]}")
plt.show()

data = spike_mem[0][2]
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(time*sr), data[2], label="facilitatory")
ax[0].plot(np.arange(time*sr), data[3], label="trigger")
ax[0].legend()
ax[0].set_title("TDE recording portion")

ax[1].plot(np.arange(time*sr), data[0], label="membrane")
ax[1].plot(np.arange(time*sr), data[1], label="spike")
ax[1].legend()
ax[1].set_title("Membrane and Spiking behaviour")

fig.suptitle(f"Spiking behaviour for SIM: frequency={frequency[2]}, angle={angles[0]}")
plt.show()

# compare tde by angle
data = spike_mem[0][0]
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(time*sr), data[2], label="facilitatory")
ax[0].plot(np.arange(time*sr), data[3], label="trigger")
ax[0].legend()
ax[0].set_title("TDE recording portion")

ax[1].plot(np.arange(time*sr), data[0], label="membrane")
ax[1].plot(np.arange(time*sr), data[1], label="spike")
ax[1].legend()
ax[1].set_title("Membrane and Spiking behaviour")

fig.suptitle(f"Spiking behaviour for SIM: frequency={frequency[0]}, angle={angles[0]}")
plt.show()

data = spike_mem[2][0]
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(time*sr), data[2], label="facilitatory")
ax[0].plot(np.arange(time*sr), data[3], label="trigger")
ax[0].legend()
ax[0].set_title("TDE recording portion")

ax[1].plot(np.arange(time*sr), data[0], label="membrane")
ax[1].plot(np.arange(time*sr), data[1], label="spike")
ax[1].legend()
ax[1].set_title("Membrane and Spiking behaviour")

fig.suptitle(f"Spiking behaviour for SIM: frequency={frequency[0]}, angle={angles[2]}")
plt.show()
# generate graphs for ITD
fig_time = plt.figure()
ax = fig_time.add_subplot(111)
color = iter(cm.rainbow(np.linspace(0, 1, len(frequency))))  # create unique colors for each plot




