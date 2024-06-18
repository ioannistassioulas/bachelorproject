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
sr = 24000

# start parsing through csv file and saving information
for file in metadata_tau:  # parse through each audio event
    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    audio = audio_tau_loc + root_name + ".wav"

    # process audio file and record to other array
    dummy, intensities = wavfile.read(audio)

    # include event plots of before and after downscaling to check for information loss
    plt.eventplot(intensities[0])
    samp = int(len(intensities) * sr / dummy)
    intensities = signal.resample(intensities, samp).transpose()
    plt.eventplot(intensities[0])
    plt.show()

    stereo.append(intensities)

    # start reading through specific csv file
    meta_csv = pd.read_csv(meta)

    # create recording arrays to extract information from csv (record of all sound events in file)
    start = []
    end = []
    angle = []
    distance = []
    elevation = []

    for i in meta_csv.transpose():  # parse through each sound event
        sound = np.array(meta_csv.loc[i])

        # append all relevant information
        start.append(sound[1])
        end.append(sound[2])
        elevation.append(sound[3])
        angle.append(sound[4])
        distance.append(sound[5])

    # add onto keypoints mega-array
    file_info = [start, end, elevation, angle, distance]
    keypoints.append(np.array(file_info).transpose())

# print complete message to timestamp
print(f"Reading data done! Time elapsed = {t.time() - start_time}s")

# START LOOPING THROUGH FILES AND EVENTS

spk_number = []
for i in range(len(metadata_tau)):  # start looking at each .wav file
    print(len(keypoints[i]))  # go through all soundevents
    for j in range(1):
        start = keypoints[i][j][0]
        end = keypoints[i][j][1]
        elevation = keypoints[i][j][2]
        angle = keypoints[i][j][3]
        distance = keypoints[i][j][4]

        # Practice filtering out all the data
        sig = [stereo[i][0], stereo[i][2]]
        timespan = [np.arange(len(np.array(sig[0]))), np.arange(len(np.array(sig[1])))]

        # look at events
        timespan, waves = audio_processing.fix_broadcasting(timespan, sig)
        print(f"Broadcasting complete! Time elapsed = {t.time() - start_time}s")
        timespan, waves = audio_processing.set_recording(timespan, waves, int(start * sr),
                                                         int(end * sr))
        print(f"Recording complete! Time elapsed = {t.time() - start_time}s")

        # start filtering sound waves
        unfiltered = waves
        time_unfiltered = timespan

        # To determine frequency band, FFT and find the strongest frequency peaks
        wave_fft = fft.fft(waves)  # peaks of fft transform
        freq_fft = fft.fftfreq(len(timespan[0]), 1 / sr)  # frequencies to check over

        l_max = np.max(wave_fft[0])
        r_max = np.max(wave_fft[1])
        main_freq_l = np.abs(freq_fft[wave_fft[0].argmax()])  # main frequency
        main_freq_r = np.abs(freq_fft[wave_fft[1].argmax()])  # main frequency

        # plot out the frequencies you need
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].plot(freq_fft, wave_fft[0], color="black")
        ax[0].scatter(main_freq_l, l_max, color="red", label=f"Primary Frequency = {main_freq_l}")
        ax[1].plot(freq_fft, wave_fft[1], color="black")
        ax[1].scatter(main_freq_r, r_max, color="red", label=f"Primary Frequency = {main_freq_r}")
        fig.text(0.5, 0.04, 'Frequency', ha='center')
        fig.text(0.04, 0.5, 'Fourier transform of wave', va='center', rotation='vertical')
        fig.suptitle("Frequency spectrum of wave obtained via FFT")
        ax[0].legend()
        ax[1].legend()
        plt.show()

        avg_freq = int(np.mean([main_freq_l, main_freq_r]))
        freq_band = [avg_freq - 5, avg_freq + 5]

        waves = audio_processing.filter_waves(waves, freq_band, "bandpass")
        print(f"Filtering complete! Time elapsed = {t.time() - start_time}s")

        # sanity check to make sure waves still look good
        wave_fft = fft.fft(waves)  # peaks of fft transform
        freq_fft = fft.fftfreq(len(timespan[0]), 1 / sr)  # frequencies to check over

        l_max = np.max(wave_fft[0])
        r_max = np.max(wave_fft[1])
        main_freq_l = np.abs(freq_fft[wave_fft[0].argmax()])  # main frequency
        main_freq_r = np.abs(freq_fft[wave_fft[1].argmax()])  # main frequency

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].plot(freq_fft, wave_fft[0], color="black")
        ax[0].scatter(main_freq_l, l_max, color="red", label=f"Primary Frequency = {main_freq_l}")
        ax[1].plot(freq_fft, wave_fft[1], color="black")
        ax[1].scatter(main_freq_r, r_max, color="red", label=f"Primary Frequency = {main_freq_r}")
        fig.text(0.5, 0.04, 'Frequency', ha='center')
        fig.text(0.04, 0.5, 'Fourier transform of wave', va='center', rotation='vertical')
        fig.suptitle("Frequency spectrum of wave obtained via FFT")
        ax[0].legend()
        ax[1].legend()
        plt.show()

        # calculate all zero crossings
        zero_x, zero_y = audio_processing.zero_crossing(waves, sr)
        zero_x[0] = zero_x[0] + start  # fix problems with zeros and linmes crossing on the bar
        zero_x[1] = zero_x[1] + start
        zero_x, zero_y = audio_processing.fix_broadcasting(zero_x, zero_y)
        print(f"Zero crossing complete! Time elapsed = {t.time() - start_time}s")

        # plot to show effectiveness of filtering
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(np.array(time_unfiltered) / sr, unfiltered, color="red")
        ax[0].set_title("Unfiltered sound event")

        ax[1].plot(timespan[0] / sr, waves[0], color="blue", label="left")
        ax[1].scatter(zero_x[0], [0] * len(zero_x[0]), color="black", label="zero crossings")
        ax[1].plot(timespan[1] / sr, waves[1], color="green", label="right")
        ax[1].scatter(zero_x[1], [0] * len(zero_x[1]), color="grey", label="zero crossings")
        ax[1].set_title("Filtered sound event")

        fig.suptitle(f"Effect of filtering soundwaves, bandpass of {freq_band} Hz. Split 1, Event 1")
        plt.legend()
        plt.show()

        print(f"First graph complete! Time elapsed = {t.time() - start_time}s")

        time_differences = np.abs(zero_x[1] - zero_x[0])  # experimental phase crossing
        inter_distance = 2 * np.sin(np.deg2rad(35)) * 0.042
        plt.plot(np.arange(len(time_differences)), time_differences, color="red",
                 label=f"expected {(45 - angle)}")  # The final degrees are taken as (90 - keypoints[0][2][0]) - 45
        plt.title(f"ITD, calculated angle = {np.rad2deg(stats.mode(ap.angle_itd(inter_distance, time_differences))[0])}")

        plt.xlabel("Time")
        plt.ylabel("recorded phase difference")
        plt.legend()
        plt.show()
        print(f"Completed angle calculation! Time elapsed = {t.time() - start_time}s")

        # transfer zeros into spike train
        zero_x[0] = (zero_x[0] - start) * sr  # subtract by start time such that length is consistent with timestep length
        current_f = torch.zeros(len(timespan[0]))
        for j in zero_x[0]:
            current_f[int(j)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        # repeating for trigger input
        zero_x[1] = (zero_x[1] - start) * sr
        current_t = torch.zeros(len(timespan[1]))
        for j in zero_x[1]:
            current_t[int(j)] = torch.ones(1)

        # pass spike train into tde simulator

        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.005)
        timer = int((end - start) * sr)

        mem, spk, fac, trg = tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(timer - 1), current_f, current_t)

        # plot spiking behaviour of first part
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.arange(timer - 1), fac[0], label="facilitatory")
        ax[0].plot(np.arange(timer - 1), trg[0], label="trigger")
        ax[0].legend()
        ax[0].set_title("TDE recording portion")

        ax[1].plot(np.arange(timer - 1), mem[0], label="membrane")
        ax[1].plot(np.arange(timer - 1), spk[0], label="spike")
        ax[1].legend()
        ax[1].set_title("Membrane and Spiking behaviour")

        fig.suptitle("Spiking behaviour for Split 1, Event 1")
        plt.show()

        # start counting all spikes
        spk_grp_count = len(torch.unique_consecutive(spk[0])) / 2
        spike_count = torch.sum(spk[0])
        avg_spk_count = spike_count / spk_grp_count

        # add spike count to final counter alongside the required angle
        angle = 45 - angle
        results = [spike_count, angle, elevation, distance, i+1, j+1]
        spk_number.append(results)

    # complete! onto the next sound event
