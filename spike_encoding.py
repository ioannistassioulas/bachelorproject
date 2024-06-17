import audio_processing
from audio_processing import *

samplecount = 2
# metadata involves array of all info in the order of angles, frequency to parse through in
# the function calls, and then ground truth time difference to compare,
# followed  by finally the calculated time difference from the simulations and the
# level difference in the simulations
metadata = np.array([[15, 30, 45, 60, 75], [200, 500, 800, 2000, 5000, 8000],
                     [0.000844834, 0.000757457, 0.000618461, 0.000437318, 0.000226372],
                     np.zeros(5 * 6), np.zeros(5 * 6), ["red", "blue", "orange", "green", "black", "brown"]],
                    dtype=object)

# determine the directory location wrt current working directory
home = os.getcwd()
database = home + "/datasets/audio files/"
database_gt = home + "/datasets/ground truth"

location = [None] * samplecount
filename = [None] * samplecount
avg = [None] * samplecount
avg_1 = [None] * samplecount
ratio = [None] * samplecount

# read pickle files inside of data
for i in range(samplecount):
    file = database_gt + f"/ssl-data_2017-04-29-13-41-12_{i}.gt.pkl"
    df = pd.read_pickle(file)
    filename[i] = database + df[0] + ".wav"
    location[i] = np.array(df[3][0][0])

angle = [np.rad2deg(np.arctan(location[i][1]/location[i][0])) for i in range(samplecount)]
colors = list(mcolors.XKCD_COLORS)
cutoff = 0
endtime = 10

for i in range(len(location)):
    # location of microphones, use channel 1 for left ear and channel 3 for right ear
    left_ear = np.array([-0.0267, 0.0343, 0])
    right_ear = np.array([0.0313, 0.0343, 0])

    # developing array of angle ground truths to compare
    inter_aural_distance = np.abs(left_ear - right_ear)[0]  # keep inter aural distance recorded
    # location = location - np.array([0, 0.0343, 0])  # adjust location for sound source

    # download sound files and separate into spike datafile
    # stereo, sampling_rate = librosa.load(filename[i], mono=False)
    waves = [stereo[0], stereo[2]]

    # create data for ITD/ILD
    spike_data, spike_values = audio_processing.zero_crossing(waves, sampling_rate)  # generate spike data values
    spike_x, spike_y = audio_processing.peak_difference(waves, sampling_rate)

    # fix any broadcasting issues
    spike_data, spike_values = audio_processing.fix_broadcasting(spike_data, spike_values)
    spike_x, spike_y = audio_processing.fix_broadcasting(spike_x, spike_y)

    # adjust the start and stop time of the recording
    spike_data, spike_values = audio_processing.set_recording(spike_data, spike_values, cutoff, endtime)
    spike_x, spike_y = audio_processing.set_recording(spike_x, spike_y, cutoff, endtime)

    # determine the inter aural time difference from the data amassed
    time_difference = np.abs(spike_data[1] - spike_data[0])  # find difference in zero crossings from both channels
    avg[i] = np.mean(time_difference)
    angle_rad = np.rad2deg(audio_processing.angle_itd(inter_aural_distance, time_difference, 343))
    non_broken = angle_rad[~np.isnan(angle_rad)]

    level_difference = np.abs(spike_y[0] - spike_y[0])
    avg_1[i] = np.mean(level_difference)

    # create plots
    plt.scatter(spike_data[0], time_difference, color=colors[i+10])
    truth_check = inter_aural_distance * np.cos(angle[i]) / 343
    plt.axhline(truth_check, color=colors[i+10], label=f"truth check of {truth_check} for {angle[i]}")


plt.xlabel("Time")
plt.ylabel("ITD")
plt.title(f"Interaural time difference (ITD) as a function of time from {cutoff}s to {endtime}s")
plt.legend(loc="upper left")
plt.show()

plt.plot(angle, avg)
plt.xlabel("angles")
plt.ylabel("ITD average")
plt.title("average ITD against angle")
plt.show()

plt.plot(angle, ratio)
plt.xlabel("angles")
plt.ylabel("ILD average")
plt.title("average ILD against angle")
plt.show()


