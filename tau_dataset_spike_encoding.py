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

    # create recording arrays to extract information from csv
    start = []
    end = []
    angle = []
    distance = []

    for i in meta_csv.transpose():  # parse through all sound events
        sound = np.array(meta_csv.loc[i])

        # append all relevant information
        start.append(sound[1])
        end.append(sound[2])
        angle.append(sound[3])
        distance.append(sound[5])

    # add onto keypoints mega-array
    file_info = [start, end, angle, distance]
    keypoints.append(file_info)

# print complete message to timestamp
print(f"Reading data done! Time elapsed = {t.time() - start_time}s")

# Practice filtering out all the data
sig = stereo[0]
waves = audio_processing.filter_waves(stereo[0], band="bandpass")
filtered_left = waves[0]
filtered_right = waves[1]

fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(len(sig[0])), sig[0], color="red")
ax[0].set_title("Unfiltered sound event")
ax[1].plot(np.arange(len(filtered_left)), filtered_left, color="blue")
ax[1].set_title("Filtered sound event")

fig.suptitle("Effect of filtering soundwaves, bandpass of [0.4, 0.5]")
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

