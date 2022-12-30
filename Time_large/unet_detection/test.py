import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import tensorflow as tf
import util
import dataset
from config import *
import model
import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt


# multi gpu scope
multi_gpu_strategy = tf.distribute.MirroredStrategy()
with multi_gpu_strategy.scope():
    # model initialize
    unet = model.UNet(16)
    detection = model.Detection()

    frame_size = past_size + present_size + future_size


    @tf.function
    def test_step(dist_inputs):
        return_list = []
        def step_fn(inputs):
            index, x, y_true = inputs
            if len(x.shape) == 0:
                x = tf.zeros([0, frame_size])
                y_true = tf.zeros([0, frame_size])

            y_pred = tf.squeeze(unet(tf.expand_dims(x, 2)), 2)
            predictions = tf.round(detection(tf.expand_dims(x, 2), tf.expand_dims(y_pred, 2)))
            labels = tf.expand_dims(tf.where(tf.reduce_sum(tf.abs(y_true - x), axis=1) > 1e-5, 1., 0.), axis=1)

            predictions = tf.where(labels == 0., 0., predictions)
            out = predictions * y_pred + (1 - predictions) * x
            # out = labels * y_pred + (1 - labels) * x

            wrong_cnt = tf.reduce_sum(tf.abs(tf.squeeze(labels, axis=1) - tf.squeeze(predictions, axis=1)))
            correct_cnt = x.shape[0] - wrong_cnt

            local_bath_size = index.shape[0]
            if local_bath_size != 0:
                out_list = tf.split(out, num_or_size_splits=local_bath_size, axis=0)
                x_list = tf.split(x, num_or_size_splits=local_bath_size, axis=0)
                predictions_list = tf.split(tf.ones_like(y_pred) * predictions, num_or_size_splits=local_bath_size, axis=0)
                labels_list = tf.split(labels, num_or_size_splits=local_bath_size, axis=0)
                for i in range(local_bath_size):
                    return_list.append([index[i], tf.squeeze(out_list[i]), tf.squeeze(x_list[i]), predictions_list[i], labels_list[i]])
            return tf.convert_to_tensor([wrong_cnt, correct_cnt])

        per_loss = multi_gpu_strategy.run(step_fn, args=(dist_inputs,))
        multi_gpu_loss = multi_gpu_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=None)
        return return_list, multi_gpu_loss


    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # load model
    if load_checkpoint_name != "":
        unet_load_checkpoint_path = os.path.join('./checkpoint', unet_load_checkpoint_name)
        unet.load_weights(os.path.join(unet_load_checkpoint_path, 'data.ckpt'))

        load_checkpoint_path = os.path.join('./checkpoint', load_checkpoint_name)
        detection.load_weights(os.path.join(load_checkpoint_path, 'detection_data.ckpt'))
    else:
        util.raise_error("Check load_checkpoint_name.")

    # read file list
    source_file_list = util.read_path_list(test_source_path, 'wav')
    target_file_list = util.read_path_list(test_target_path, 'wav')

    # dataset initialize
    total_wrong_cnt, total_correct_cnt = 0, 0
    for i, (source_file, target_file) in enumerate(zip(source_file_list, target_file_list)):
        test_dataset = dataset.AudioDataset(audio_file_path=[source_file, target_file],
                                             sampling_rate=sampling_rate, frame_size=frame_size, shift_size=shift_size,
                                             batch_size=batch_size, shuffle=False,
                                             window_type='sine')
        dist_test_dataset = multi_gpu_strategy.experimental_distribute_dataset(test_dataset.get_dataset().with_options(options))

        save_file_name = util.remove_base_path(source_file, test_source_path)
        save_file_path = os.path.join('./test_result', load_checkpoint_name, save_file_name)

        # run
        start = time.time()
        now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        num_of_iteration = math.ceil(test_dataset.get_number_of_frame()/batch_size)
        return_list = []
        for iteration, inputs in enumerate(dist_test_dataset):
            print(f"\r({now_time})", end=" ")
            print(util.change_font_color('bright cyan', 'Test:'), end=" ")
            print(util.change_font_color('yellow', 'file'), util.change_font_color('bright yellow', f"({i}/{len(source_file_list)}) {save_file_name},"), end=" ")
            print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{iteration+1}/{num_of_iteration}"), end=" ")
            temp_return_list, temp_cnt = test_step(inputs)
            return_list.extend(temp_return_list)
            total_wrong_cnt += temp_cnt[0]
            total_correct_cnt += temp_cnt[1]
            print(f'{total_correct_cnt} / {total_wrong_cnt}')

        # output
        orig_pred_output = np.zeros(test_dataset.get_total_length())
        orig_output = np.zeros(test_dataset.get_total_length())
        pred_output = np.zeros(test_dataset.get_total_length())
        label_output = np.zeros(test_dataset.get_total_length())

        for index, orig_pred, orig, prediction, label in return_list:
            index = int(index)
            orig_pred = orig_pred.numpy()
            orig_pred_output[index*shift_size+past_size:index*shift_size+past_size+frame_size] += orig_pred * util.window('sine', present_size)
            # orig_output[index*shift_size+past_size:index*shift_size+past_size+frame_size] += orig * util.window('sine', present_size)
            # pred_output[index*shift_size+past_size:index*shift_size+past_size+frame_size] += prediction * util.window('hanning', present_size)
            # label_output[index*shift_size+past_size:index*shift_size+past_size+frame_size] += label * util.window('hanning', present_size)

        orig_pred_output = orig_pred_output[test_dataset.get_padding_size()[0]:len(orig_pred_output)-test_dataset.get_padding_size()[1]]
        # orig_output = orig_output[test_dataset.get_padding_size()[0]:len(orig_output) - test_dataset.get_padding_size()[1]]
        # pred_output = pred_output[test_dataset.get_padding_size()[0]:len(pred_output) - test_dataset.get_padding_size()[1]]
        # label_output = label_output[test_dataset.get_padding_size()[0]:len(label_output) - test_dataset.get_padding_size()[1]]
        #
        # plt.plot(orig_pred_output)
        # plt.plot(orig_output)
        # plt.plot(pred_output * max(orig_pred_output))
        # plt.plot(label_output * max(orig_pred_output))
        #
        # fig_path = f'./save_fig/{save_file_name.replace(".wav", ".png")}'
        # os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        # plt.savefig(fig_path)
        # plt.close()

        util.write_audio_file(save_file_path, orig_pred_output, sampling_rate)
        # util.write_audio_file(save_file_path.replace(".wav", "_orig.wav"), orig_output, sampling_rate)
        # util.write_audio_file(save_file_path.replace(".wav", "_pred.wav"), pred_output, sampling_rate)

        end_time = util.second_to_dhms_string(time.time()-start)
        print(f"({end_time})")
    accuracy = total_correct_cnt / (total_correct_cnt + total_wrong_cnt)
    print(f"accuracy = {accuracy}")
