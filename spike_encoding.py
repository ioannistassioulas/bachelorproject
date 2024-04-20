import audio_processing
from audio_processing import *

stereo, sampling_rate = librosa.load("/home/s4290755/Downloads/sample_sound_data.wav", mono=False)

print(audio_processing.count_peak(stereo))
