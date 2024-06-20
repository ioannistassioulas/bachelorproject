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
sr = 1000
# start parsing through csv file and saving information
for file in metadata_tau:  # parse through each audio event
    # get names of files
    root_name = file.partition('.')[0]
    meta = metadata_tau_loc + root_name + ".csv"
    audio = audio_tau_loc + root_name + ".wav"

    # process audio file and record to other array
    dummy, intensities = wavfile.read(audio)
    samp = int(len(intensities) * sr / dummy)

    intensitiesBefore = np.array([intensities.transpose()[0], intensities.transpose()[2]])
    intensitiesAfter = np.array([signal.resample(intensitiesBefore[0], samp), signal.resample(intensitiesBefore[1], samp)])
    a = {"IBL": intensitiesBefore[0], "IBR": intensitiesBefore[1],
                              "IAL": intensitiesAfter[0], "IAR": intensitiesAfter[1]}
    stereoCSV = pd.DataFrame.from_dict(a, orient='index').transpose()
    stereoCSV.to_csv(f"results/{root_name}-down-sampling.csv")

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
for i in range(len(metadata_tau)):  # start looking at each .wav file
    root_name = metadata_tau[i].partition('.')[0]
    stereo = pd.read_csv(f"results/{root_name}-down-sampling.csv")
    sig = [stereo['IAL'].to_numpy(), stereo['IAR'].to_numpy()]
    sig = [sig[0][~np.isnan(sig[0])], sig[1][~np.isnan(sig[1])]]
    print(sig)
    for j in range(1):  # go through all sound events

        start = keypoints[i][j][0]
        end = keypoints[i][j][1]
        elevation = keypoints[i][j][2]
        angle = keypoints[i][j][3]
        distance = keypoints[i][j][4]

        # Practice filtering out all the data
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

        # header : frequency range, left ear fft, right ear fft, left peak x, left peak y, right peak x, right peak y
        b = {"Freq-Range": freq_fft, "Left-FFT": wave_fft[0], "Right-FFT": wave_fft[1]}

        freq_info = pd.DataFrame.from_dict(b, orient="index").transpose()
        freq_info.to_csv(f"results/Fourier-Results-Split-{i+1}-Event-{j+1}.csv")
        avg_freq = int(np.mean([main_freq_l, main_freq_r]))
        freq_band = [avg_freq - 5, avg_freq + 5]

        waves = audio_processing.filter_waves(waves, freq_band, "bandpass")
        print(f"Filtering complete! Time elapsed = {t.time() - start_time}s")

        # calculate all zero crossings
        zero_x, zero_y = audio_processing.zero_crossing(waves, sr)
        zero_x[0] = zero_x[0] + start  # fix problems with zeros and linmes crossing on the bar
        zero_x[1] = zero_x[1] + start
        zero_x, zero_y = audio_processing.fix_broadcasting(zero_x, zero_y)

        c = {"ZL": zero_x[0], "ZR": zero_x[1], "WL": waves[0], "WR": waves[1]}
        zero_data = pd.DataFrame.from_dict(c, orient="index").transpose()
        zero_data.to_csv(f"results/zero_crossings_data_Split{i+1}_Split{j+1}.csv")
        print(f"Zero crossing complete! Time elapsed = {t.time() - start_time}s")

        # delete all time intervals that are higher than expected (due to fault)
        time_differences = np.abs(zero_x[1] - zero_x[0])  # experimental phase crossing
        inter_distance = 2 * np.sin(np.deg2rad(35)) * 0.042
        final_angle = np.rad2deg(ap.angle_itd(inter_distance, time_differences))

        # throw away to try and save a bit of memory
        gc.collect()

        # transfer zeros into spike train
        zero_x[0] = (zero_x[0] - start) * sr  # subtract by start time such that length is consistent with timestep length
        current_f = torch.zeros(len(timespan[0]))
        for j1 in zero_x[0]:
            current_f[int(j1)] = torch.ones(1)  # at each index of the time step you input in facilitatory array

        # repeating for trigger input
        zero_x[1] = (zero_x[1] - start) * sr
        current_t = torch.zeros(len(timespan[1]))
        for j2 in zero_x[1]:
            current_t[int(j2)] = torch.ones(1)

        # pass spike train into tde simulator

        tau_tde = torch.tensor(0.001)
        tau_mem = torch.tensor(0.005)
        timer = int((end - start) * sr)

        mem, spk, fac, trg = tde(tau_tde, tau_tde, tau_mem, torch.tensor(1/sr), torch.tensor(timer - 1), current_f, current_t)
        tde_results = pd.DataFrame([mem[0], spk[0], fac[0], trg[0]])
        tde_results.to_csv(f"results/LIF-Response-Split{i+1}-Event-{j+1}.csv")

        # start counting all spikes
        total_spike = torch.tensor(signal.convolve(spk[0], mem[0]))
        spike_count = torch.sum(total_spike)

        # add spike count to final counter alongside the required angle
        angle = 45 - angle
        results = pd.DataFrame([spike_count, angle, elevation, distance])
        results.to_csv(f"results/final_split_{i+1}-event_{j+1}.csv")
        
        # throw away to try and save a bit of memory
        gc.collect()

# l_max = np.max(wave_fft[0])
# r_max = np.max(wave_fft[1])
# main_freq_l = np.abs(freq_fft[wave_fft[0].argmax()])  # main frequency
# main_freq_r = np.abs(freq_fft[wave_fft[1].argmax()])  # main frequency