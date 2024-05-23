import audio_processing as ap
from audio_processing import *

import snnTorch_tutorial as sn
from snnTorch_tutorial import *
# get current directory
home = os.getcwd()
one_up = path.abspath(path.join(home, ".."))
metadata_tau_loc = one_up + "/metadata_dev/"
audio_tau_loc = one_up + "/mic_dev/"

# retrieve all filenames
metadata_tau = os.listdir(metadata_tau_loc)

# array of all files most important info
# setup of array : [audio file, sound_event, metadata of event]
keypoints = []
stereo = []
sr = 0
k = 1

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
        print(f"Running . . . Iteration number {k}")
        k += 1

    # add onto keypoints mega-array
    file_info = [start, end, angle, distance]
    keypoints.append(file_info)


# start looking into the keypoints mega-array
print("Done!")

time_difference = []
level_difference = []
time = []
index = 0
for file in keypoints:

    # select sound_file to analyze
    sound_data = stereo[index]

    waves = [sound_data[0], sound_data[2]]

    # generate ITD/ILD data of that subset
    itd_x, itd_y = ap.zero_crossing(waves, sr)
    ild_x, ild_y = ap.peak_difference(waves, sr)

    # fix any issues in broadcasting
    itd_x, itd_y = ap.fix_broadcasting(itd_x, itd_y)
    ild_x, ild_y = ap.fix_broadcasting(ild_x, ild_y)

    for split in file:
        # import metadata for ground-truth comparison
        start = split[0]
        end = split[1]
        ground_truth_theta = split[2]
        ground_truth_distance = split[3]

        # set recording to analyze only that one sound event
        xt, yt = ap.set_recording(itd_x, itd_y, start, end)
        xl, yl = ap.set_recording(ild_x, ild_y, start, end)

        # append critical values into time difference array
        time_difference.append([xt, yt])
        level_difference.append([xl, yl])
        time.append(end-start)
    # iterate the index of the sound_file parser
    index += 1

print("zero-encoding complete!")

# start sending data in to run for spikes
for i in range(len(time)):
    #
    timespan = time[i]
    facilitatory_time, trigger_time = time_difference[i]
    facilitatory_level, trigger_level = level_difference[i]

    post_synaptic, i, v, spikes = sn.tde_model(timespan, facilitatory_time, trigger_time, 1e20, 1e20, sr)

    plt.plot()
