import audio_processing as ap
from audio_processing import *

import neural_network_model as sn
from neural_network_model import *

import time as t
# start recording time
start_time = t.time()

# get current directory
home = os.getcwd()
one_up = path.abspath(path.join(home, ".."))
metadata_tau_loc = one_up + "/metadata_dev/"
audio_tau_loc = one_up + "/mic_dev/"

# retrieve all filenames
metadata_tau = os.listdir(metadata_tau_loc)
print(f"Acquiring sound files done! Time elapsed = {t.time() - start_time}s")

# array of all files most important info
# setup of array : [audio file, sound_event, metadata of event]
keypoints = []
stereo = []
sr = 48000

# start parsing through csv file and saving information
for file in metadata_tau:  # parse through each audio split
    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    audio = audio_tau_loc + root_name + ".wav"

    # process audio file and record to other array
    intensities, sampling_rate = librosa.load(audio, mono=False, sr=None)  # REPLACE THIS WITH SCIPY
    stereo.append(intensities)
    sr = sampling_rate

    # start reading through specific csv file
    meta_csv = pd.read_csv(meta)

    # create recording arrays to extract information from csv (record of all sound events in file)
    start = []
    end = []
    angle = []
    distance = []
    elevation = []

    for i in meta_csv.transpose():  # parse through all sound events
        sound = np.array(meta_csv.loc[i])

        # append all relevant information
        start.append(sound[1])
        end.append(sound[2])
        elevation.append(sound[3])
        angle.append(sound[4])
        distance.append(sound[5])

    # add onto keypoints mega-array
    file_info = [start, end, elevation, angle, distance]
    keypoints.append(file_info)


# print complete message to timestamp
print(f"Reading data done! Time elapsed = {t.time() - start_time}s")

# Practice filtering out all the data
sig = [stereo[0][0], stereo[0][2]]
timespan = [np.arange(len(np.array(sig[0]))), np.arange(len(np.array(sig[1])))]

# fix all broadcasting issues
timespan, waves = audio_processing.fix_broadcasting(timespan, sig)
print("completed broadcasting")
timespan, waves = audio_processing.set_recording(timespan, waves, int(keypoints[0][0][0] * sr), int(keypoints[0][1][0] * sr))
print("complete recording set")

# apply final filter and count zeros
unfiltered = waves
time_unfiltered = timespan
waves = audio_processing.filter_waves(waves, band="bandpass")
zero_x, zero_y = audio_processing.zero_crossing(waves, sr, int(keypoints[0][0][0] * sr))

# add start time such that zeros and wave line up on graph
zero_x[0] = zero_x[0] + keypoints[0][0][0]
zero_x[1] = zero_x[1] + keypoints[0][0][0]

#realign all broadcasting issues again
zero_x, zero_y = audio_processing.fix_broadcasting(zero_x, zero_y)

# plot to show effectiveness of filtering
fig, ax = plt.subplots(3, 1)

ax[0].plot(np.array(time_unfiltered) / sr, unfiltered, color="red")
ax[0].set_title("Unfiltered sound event")

ax[1].plot(timespan[0] / sr, waves[0], color="blue")
ax[1].scatter(zero_x[0], [0] * len(zero_x[0]), color="black", label="zero crossings")
ax[1].set_title("Filtered sound event, left")

ax[2].plot(timespan[1] / sr, waves[1], color="green")
ax[2].scatter(zero_x[1], [0] * len(zero_x[1]), color="black", label="zero crossings")
ax[2].set_title("Filtered sound event, right")

fig.suptitle("Effect of filtering soundwaves, bandpass of [0.2, 0.25]. Split 1, Event 1")
plt.legend()
plt.show()

# The final degrees are taken as keypoints[0][2][0] + 45 - 90 -> keypoints[0][3][0] - 45

time_differences = np.abs(zero_x[1] - zero_x[0])  # experimental phase crossing
print(f"final answer is {np.mean(time_differences)} with a standard deviation of {np.std(time_differences)}, error of {np.mean(time_differences) - np.std(time_differences) / np.mean(time_differences)}")
plt.plot(np.arange(len(time_differences)), time_differences, color="red")
inter_distance = 2 * np.sin(np.deg2rad(35)) * 0.042
plt.axhline(audio_processing.angle_itd(inter_distance, np.mean(time_differences)), color="blue")
plt.show()

# create memory arrays to hold all itd and ild values
time_difference = []
level_difference = []
times = []
gt = []

# # index to parse through file
# index = 0
# for file in keypoints:
#     timestamp = t.time()
#
#     # select sound_file to analyze
#     sound_data = stereo[index]
#     waves = audio_processing.filter_waves(sound_data, band="bandpass")
#
#     # generate ITD/ILD data of that subset
#     itd_x, itd_y = ap.zero_crossing(waves, sr)
#     ild_x, ild_y = ap.peak_difference(waves, sr)
#
#     fig, ax = plt.subplots(2, 1)
#     ax[0].plot(np.arange(len(waves[0])), waves[0])
#     ax[1].plot(np.arange(len(waves[1])), waves[1])
#     ax[0].scatter(itd_x[0] * sr, itd_y[0])
#     ax[1].scatter(itd_x[1] * sr, itd_y[1])
#     plt.show()
#
#     jindex = 0
#     for split in np.array(file).transpose():
#
#         # import metadata for ground-truth comparison
#         start = split[0]
#         end = split[1]
#         ground_truth_theta = split[2]
#         ground_truth_distance = split[3]
#
#         # set recording to analyze only that one sound event
#         xt, yt = ap.set_recording(itd_x, itd_y, start, end)
#         xl, yl = ap.set_recording(ild_x, ild_y, start, end)
#
#         # fix any issues in broadcasting
#         xt, yt = ap.fix_broadcasting(xt, yt)
#         xl, yl = ap.fix_broadcasting(xl, yl)
#
#         # append critical values into time difference array
#         time_difference.append(xt)
#         level_difference.append(yt)
#         times.append([start, end])
#         gt.append([ground_truth_theta, ground_truth_distance, f"sound file number {index + 1}, sound event number {jindex + 1}, timespan = {start}s to {end}s"])
#         jindex += 1
#
#     # iterate the index of the sound_file
#     print(f"Split number {index + 1} done! Time elapsed = {t.time() - timestamp}s")
#     index += 1
#
# # timestamp to find out when zero encoding is done
# print(f"zero-encoding complete! Total time elapse = {t.time() - start_time}s")

# Begin looking at the spikerinos

# transfer zeros into spike train
zero_x[0] = (zero_x[0] - start) * sr  # subtract by start time such that length is consistent with timestep length
current_f = torch.zeros(timespan)
for j in zero_x[0]:
    current_f[int(j)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

# repeating for trigger input
zero_x[1] = (zero_x[1] - start) * sr
current_t = torch.zeros(timespan)
for j in zero_x[1]:
    current_t[int(j)] = torch.ones(1)

