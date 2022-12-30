import os
import util


def calc_file_time(folder_path, sampling_rate):
    file_path_list = util.read_path_list(folder_path, extension='wav')

    sample_length = 0
    for file_path in file_path_list:
        sample_length += len(util.read_audio_file(file_path, sampling_rate=sampling_rate))
    print(util.second_to_dhms_string(sample_length / sampling_rate))


folder_path = "/home/ubuntu/SaturationEnhancement/subjective_file/TimeUnet"
calc_file_time(folder_path, 16000)