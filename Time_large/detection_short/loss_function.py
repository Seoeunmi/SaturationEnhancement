import tensorflow as tf
import util
from config import *


def binary_cross_entropy_function():
    def func(labels, predictions):
        return -(tf.reduce_sum(labels * util.safe_log(predictions) + (1 - labels) * util.safe_log(1 - predictions))) / batch_size
    return func


def mse_loss_function():
    def func(y_true, y_pred, axis=1):
        return tf.reduce_sum(tf.reduce_mean(tf.math.square(y_true-y_pred), axis=axis)) / batch_size
    return func


def magnitude_mse_loss_function():
    def func(y_true, y_pred):
        y_true_mag = util.safe_log(tf.math.abs(tf.signal.fft(tf.cast(y_true, dtype=tf.complex64))) / y_true.shape[1])
        y_pred_mag = util.safe_log(tf.math.abs(tf.signal.fft(tf.cast(y_pred, dtype=tf.complex64))) / y_true.shape[1])
        loss = tf.reduce_sum(tf.reduce_mean(tf.math.square(y_true_mag - y_pred_mag), axis=1)) / batch_size
        return loss
    return func


def multi_scale_magnitude_loss_function():
    stft_scale = [2048, 1024, 512, 256, 128, 64]

    def func(y_true, y_pred):
        loss = []
        for f in stft_scale:
            y_true_mag = 20. * util.safe_log10(tf.abs(tf.signal.stft(tf.squeeze(y_true, 2), f, f // 4)) / f)
            y_pred_mag = 20. * util.safe_log10(tf.abs(tf.signal.stft(tf.squeeze(y_pred, 2), f, f // 4)) / f)
            loss.append(tf.reduce_sum(tf.reduce_mean(tf.abs(y_true_mag - y_pred_mag), [1, 2])))
        loss = tf.add_n(loss) / batch_size / len(stft_scale)
        return loss
    return func