import audio_processing
from audio_processing import *

# create sound waves

frequency = np.linspace(250, 10000, 40)
angles = np.linspace(0, 90, 10)
sr = 48000
distance = 0.001

# create arrays to store ITD and ILD information
time_difference = np.zeros([len(angles), len(frequency)])
level_difference = np.zeros([len(angles), len(frequency)])

# see about about counter
k = 1

# go through all angles and frequencies again and apply tests
for i in range(len(angles)):  # by angle
    for j in range(len(frequency)):  # by frequency
        cutoff = 0
        endtime = 99

        l, r = audio_processing.generate_test_waves(angles[i], frequency[j], sr, 5, distance)
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

        # check to make sure of progress are still working
        print(f"{k/(len(angles) * len(frequency))*100}%")
        k += 1

# generate "predicted" angles via the theory
angle_time = audio_processing.angle_itd(distance, time_difference)
angle_level = audio_processing.angle_ild(distance, level_difference)

# generate graphs for ITD
fig_time = plt.figure()
ax = fig_time.add_subplot(111)
color = iter(cm.rainbow(np.linspace(0, 1, len(angles))))  # create unique colors for each plot

for m in range(len(angles)):
    c = next(color)
    ax.scatter(frequency, angle_time[m], color=c,  label=f"angle = {angles[m]}")
    ax.axhline(angles[m], color=c)

    ax.set_title("Performance of zero crossings method for angles")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Angle")

plt.legend()
plt.show()

# likewise for ILD
fig_level = plt.figure()
ax = fig_level.add_subplot(111)
color = iter(cm.rainbow(np.linspace(0, 1, len(angles))))  # create unique colors for each plot

for n in range(len(angles)):
    c = next(color)
    ax.scatter(frequency, angle_level[n], color=c, label=f"Angles = {angles[n]}")
    ax.axhline(angles[n], color=c)

    ax.set_title("Performance of level difference for each angle")
    ax.set_xlabel("Frequencies")
    ax.set_ylabel("ILD")

plt.legend()
plt.show()


