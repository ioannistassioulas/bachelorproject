import audio_processing
from audio_processing import *

# create sound waves
sr = 44100
angles = np.linspace(0, 180, 13)
phase = np.cos(np.deg2rad(angles)) * 0.3 / 343
print(phase)
frequency = np.linspace(320, 8000, 25)
t = np.linspace(0, 5, 5*sr)
ild_test = [None] * len(angles)

waveform_1 = np.zeros([len(angles), len(frequency)], dtype=object)
waveform_2 = np.zeros([len(angles), len(frequency)], dtype=object)

k = 1
for i in range(len(angles)):  # loop through angles
    theta = angles[i]
    radius = np.array([np.cos(theta), np.sin(theta), 0])
    left = radius + np.array([0.15, 0, 0])
    right = radius + np.array([-0.15, 0, 0])

    amp_left = 1 / left.dot(left)
    amp_right = 1 / right.dot(right)

    ild_test[i] = amp_left - amp_right

    for j in range(len(frequency)):  # loop through frequencies
        waveform_1[i][j] = amp_right * np.sin(t * frequency[j])  # left channel
        waveform_2[i][j] = amp_left * np.sin((t - phase[i]) * frequency[j])  # right channel
        print(f"{100 * k / (len(angles) * len(frequency)) }%")
        k += 1


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
        waves = np.array([waveform_1[i][j], waveform_2[i][j]])
        spike_data, spike_values = audio_processing.zero_crossing(waves, sr)
        spike_x, spike_y = audio_processing.peak_difference(waves, sr)
g
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

        #check to make sure of progress are still working
        print(f"{k/(len(angles) * len(frequency))*100}%")
        k += 1


# generate graphs for ITD
fig_time = plt.figure()
ax = fig_time.add_subplot(111)
color = iter(cm.rainbow(np.linspace(0, 1, len(angles))))  # create unique colors for each plot

for m in range(len(angles)):
    c = next(color)
    ax.scatter(frequency, time_difference[m], color=c,  label=f"angle = {angles[m]}")
    ax.axhline(phase[m], color=c)

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
    ax.scatter(frequency, level_difference[n], color=c, label=f"Angles = {angles[n]}")
    ax.axhline(ild_test[n], color=c)

    ax.set_title("Performance of level difference for each angle")
    ax.set_xlabel("Frequencies")
    ax.set_ylabel("ILD")

plt.legend()
plt.show()

# start passing numbers into spiking neural network


