import glob
import os
from typing import Union
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from math import pi

def change_font_color(color='', text=None):
    if color == "black":
        return_str = "\033[30m"
    elif color == "red":
        return_str = "\033[31m"
    elif color == "green":
        return_str = "\033[32m"
    elif color == "yellow":
        return_str = "\033[33m"
    elif color == "blue":
        return_str = "\033[34m"
    elif color == "magenta":
        return_str = "\033[35m"
    elif color == "cyan":
        return_str = "\033[36m"
    elif color == "white":
        return_str = "\033[37m"
    elif color == "bright black":
        return_str = "\033[90m"
    elif color == "bright red":
        return_str = "\033[91m"
    elif color == "bright green":
        return_str = "\033[92m"
    elif color == "bright yellow":
        return_str = "\033[93m"
    elif color == "bright blue":
        return_str = "\033[94m"
    elif color == "bright magenta":
        return_str = "\033[95m"
    elif color == "bright cyan":
        return_str = "\033[96m"
    elif color == "bright white":
        return_str = "\033[97m"
    else:
        return_str = "\033[0m"
    if text:
        return_str += text + "\033[0m"
    return return_str


def raise_error(message):
    if message.find('E: ')==-1:
        message = "\nE: " + message
    raise Exception(change_font_color("bright red", message))


def raise_issue(message):
    if message.find('I: ')==-1:
        message = "\nI: " + message
    print(change_font_color("bright yellow", message))


def second_to_dhms_string(second, second_round=True):
    d, left = divmod(second, 86400)
    h, left = divmod(left, 3600)
    m, s = divmod(left, 60)
    str = ""
    d, h, m, s = int(d), int(h), int(m), int(s) if second_round else int(s) + second - int(second)
    if d!=0:
        str+=f"{d:d}d "
    if h!=0 or d!=0:
        str+=f"{h:02d}h "
    if m!=0 or h!=0 or d!=0:
        str+=f"{m:02d}m "
    str += f"{s:02d}s" if second_round else f"{s:2.2f}s"
    return str


def save_optimizer_state(optimizer, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.find('.npy') == -1:
        save_path += '.npy'
    np.save(save_path, optimizer.get_weights())


def load_optimizer_state(optimizer, load_path):
    if load_path.find('.npy') == -1:
        load_path += '.npy'
    opt_weights = np.load(load_path, allow_pickle=True)
    optimizer.set_weights(opt_weights)


def compare_path_list(file_paths: Union[list, tuple], extension=''):
    if len(file_paths) < 2:
        raise_error("Check number of file_paths in compare_path_list()")

    path_list = read_path_list(file_paths, extension)
    compare_list = []
    for i, file_path in enumerate(path_list):
        compare_list.append(list(map(lambda f: remove_base_path(f, file_paths[i]), file_path)))
    for i in range(len(compare_list)-1):
        if compare_list[i] != compare_list[i+1]:
            return False
    return True


def read_path_list(file_paths: Union[str, list, tuple], extension=''):
    extension = extension.strip('*.')
    extension_str = f'/*.{extension}' if extension != '' else '/*.*'

    if isinstance(file_paths, str):
        if os.path.isfile(file_paths):
            if os.path.splitext(file_paths)[1] != '.' + extension:
                raise_error(f'Check the extension ! -> {file_paths}')
            return [os.path.normpath(os.path.abspath(file_paths))]
        return list(map(lambda file_path: os.path.normpath(os.path.abspath(file_path)), sorted(glob.glob(file_paths + '/**' + extension_str, recursive=True))))
    else:
        file_path_list = []
        for each_path in file_paths:
            if os.path.isfile(each_path):
                if os.path.splitext(each_path)[0] != '.' + extension:
                    raise_error(f'Check the extension ! -> {each_path}')
                file_path_list.append(os.path.normpath(os.path.abspath(each_path)))
            file_path_list.append(list(map(lambda file_path: os.path.normpath(os.path.abspath(file_path)), sorted(glob.glob(each_path + '/**' + extension_str, recursive=True)))))
        return file_path_list


def remove_base_path(full_path, base_path):
    full_path = os.path.normpath(os.path.abspath(full_path))
    base_path = os.path.normpath(os.path.abspath(base_path))
    if full_path == base_path:
        return os.path.basename(full_path)
    return full_path.replace(base_path, '').lstrip('./\\')


def read_audio_file(file_path, sampling_rate=None, different_sampling_rate_detect=False):
    if different_sampling_rate_detect:
        x, sr = librosa.load(file_path, sr=None)
        if sr != sampling_rate:
            raise_error(f"Different sampling rate detected ! -> {file_path} ({sr})")
    else:
        x, _ = librosa.load(file_path, sr=sampling_rate)
    return x


def write_audio_file(file_path, data, sampling_rate):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, data, sampling_rate)


def window(window_name, frame_size, dtype='float32'):
    if window_name == 'hanning':
        window = np.hanning(frame_size)
    elif window_name == 'hamming':
        window = np.hamming(frame_size)
    elif window_name == 'sine':
        k = np.array(np.linspace(0, 1, frame_size))
        window = np.sin(np.pi * k)
    elif window_name == 'uniform':
        window = np.ones(1)
    else:
        raise_error('Select hanning, hamming, sine, uniform')
    return window.astype(dtype)


def safe_log(x, eps=1e-6):
    return tf.math.log(tf.where(x <= eps, eps, x))


def safe_log10(x, eps=1e-6):
    return tf.math.log(tf.where(x <= eps, eps, x)) / tf.math.log(10.)


def safe_angle(x, eps=1e-6):
    return tf.math.angle(tf.where(tf.math.real(x) == 0, x+eps, x))


def safe_sqrt(x, eps=1e-6):
    return tf.math.sqrt(tf.where(tf.math.real(x) == 0, x+eps, x))

def audio_log_scale_fft(x):
    return 20. * np.log10(np.abs(np.fft.fft(x))/len(x))


def custom_kbd(window_length, dtype=tf.dtypes.float32):
    kbd = tf.signal.kaiser_bessel_derived_window(window_length // 2, dtype=dtype)
    kbd_front = tf.slice(kbd, [0], [window_length // 4])
    ones = tf.ones(window_length // 2, dtype=dtype)
    kbd_back = tf.slice(kbd, [window_length // 4], [window_length // 4])
    window = tf.concat([kbd_front, ones, kbd_back], 0)
    return window


def custom_hann(window_length, dtype=tf.dtypes.float32):
    hann = tf.signal.hann_window(window_length // 2, dtype=dtype)
    hann_front = tf.slice(hann, [0], [window_length // 4])
    ones = tf.ones(window_length // 2, dtype=dtype)
    hann_back = tf.slice(hann, [window_length // 4], [window_length // 4])
    window = tf.concat([hann_front, ones, hann_back], 0)
    return window


def sine_window(window_length, dtype=tf.dtypes.float32):
    k = tf.linspace(0, 1, window_length//2)
    sin_window = tf.math.sin(pi * k)
    sin_window = tf.cast(sin_window, dtype=dtype)
    sin_front = tf.slice(sin_window, [0], [window_length // 4])
    ones = tf.ones(window_length // 2, dtype=dtype)
    sin_back = tf.slice(sin_window, [window_length // 4], [window_length // 4])
    window = tf.concat([sin_front, ones, sin_back], 0)
    return window