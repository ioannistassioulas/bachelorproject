import audio_processing
from audio_processing import *

# get current directory
home = os.getcwd()
metadata_tau_loc = home + "/datasets/tau-dataset/metadata_dev/"
audio_tau_loc = home + "/datasets/tau-dataset/mic_dev"

# retrieve all filenames
metadata_tau = os.listdir(metadata_tau_loc)

# array of all files most important info
# setup of array : [audio file, sound_event, metadata of event]
keypoints = []
stereo = []
# start parsing through csv file and saving information
for file in metadata_tau: # parse through audio file
    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + "/" + root_name + ".csv"
    audio = audio_tau_loc + "/" + root_name + ".wav"

    # process audio file and record to other array
    intensities, sampling_rate = librosa.load(audio, mono=False)
    stereo.append(intensities)

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

# start reading through all audio files


