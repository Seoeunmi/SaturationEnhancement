import os, sys
import util
import dataset
import numpy as np
import matplotlib.pyplot as plt

orig_signal_path = "/home/ubuntu/SaturationEnhancement/paper_result/p225_329_orig.wav"
clipped_signal_path = "/home/ubuntu/SaturationEnhancement/paper_result/p225_329_clipped.wav"
audition_signal_path = "/home/ubuntu/SaturationEnhancement/paper_result/p225_329_audition.wav"
unet_signal_path = "/home/ubuntu/SaturationEnhancement/paper_result/p225_329_unet.wav"
detector_signal_path = "/home/ubuntu/SaturationEnhancement/paper_result/p225_329_detector.wav"


orig_signal = util.read_audio_file(orig_signal_path, 16000)
clipped_signal = util.read_audio_file(clipped_signal_path, 16000)
audition_signal = util.read_audio_file(audition_signal_path, 16000)
unet_signal = util.read_audio_file(unet_signal_path, 16000)
detector_signal = util.read_audio_file(detector_signal_path, 16000)

with open("./plot_time.csv", "w") as f:
    f.write(f'orig,clipped,audition,unet,detector\n')
    for i in range(32000, 48000):
        f.write(f'{orig_signal[i]},{clipped_signal[i]},{audition_signal[i]},{unet_signal[i]},{detector_signal[i]}\n')

