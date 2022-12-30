import tensorflow as tf
from typing import Union
import numpy as np
import util
import os
import pandas as pd


class AudioDataset:
    def __init__(self, audio_file_path: Union[str, list, tuple], sampling_rate, frame_size, shift_size, batch_size, shuffle: Union[bool, int] = False, shuffle_seed=None, window_type='uniform', dtype='float32', extension='wav'):
        # single audio dataset
        if isinstance(audio_file_path, str):
            all_audio_path_list = [util.read_path_list(audio_file_path, extension=extension)]

        # multiple audio dataset
        else:
            if os.path.isdir(audio_file_path[0]):
                if not util.compare_path_list(audio_file_path, extension=extension):
                    util.raise_error('Audio file lists are not same')
                all_audio_path_list = util.read_path_list(audio_file_path, extension=extension)
            else:
                all_audio_path_list = [[path] for path in audio_file_path]

        # make audio data list
        self.all_audio_data_list = []
        for path_list in all_audio_path_list:
            x_list = []
            for i, path in enumerate(path_list):
                x = util.read_audio_file(path, sampling_rate=sampling_rate, different_sampling_rate_detect=False)
                x_list.append(x)
            x = np.concatenate(x_list)

            self.front_padding_size = frame_size
            self.rear_padding_size = (shift_size - (len(x) + frame_size + shift_size) % shift_size) % shift_size + frame_size
            x = np.pad(x, (self.front_padding_size, self.rear_padding_size), 'constant', constant_values=0.0).astype(dtype)
            self.number_of_frame = (len(x) - frame_size + shift_size) // shift_size
            self.total_length = len(x)
            self.all_audio_data_list.append(x)

        # check different data length
        for i in range(len(self.all_audio_data_list)-1):
            if len(self.all_audio_data_list[i]) != len(self.all_audio_data_list[i+1]):
                util.raise_error('Audio file length are not same')

        # make tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices(list(range(self.number_of_frame)))
        if isinstance(shuffle, bool):
            shuffle = self.number_of_frame if shuffle else 0
        else:
            shuffle = min(self.number_of_frame, shuffle)
        if shuffle:
            dataset = dataset.shuffle(shuffle, reshuffle_each_iteration=True, seed=shuffle_seed)
        window = util.window(window_type, frame_size, dtype)

        def func(index):
            return_list = [data[index*shift_size:index*shift_size+frame_size]*window for data in self.all_audio_data_list]
            return_list.insert(0, index)
            return return_list
        output_datatype = [tf.float32 for _ in range(len(self.all_audio_data_list))]
        output_datatype.insert(0, tf.int32)
        self.dataset = dataset.map(lambda index: tf.py_function(func=func, inp=[index], Tout=output_datatype), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        self.sampling_rate = sampling_rate

    def get_dataset(self):
        return self.dataset

    def get_number_of_frame(self):
        return self.number_of_frame

    def get_total_length(self):
        return self.total_length

    def get_padding_size(self):
        return (self.front_padding_size, self.rear_padding_size)

    def get_audio_data(self):
        return self.all_audio_data_list

    def get_sampling_rate(self):
        return self.sampling_rate

