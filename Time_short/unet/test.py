import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import tensorflow as tf
import util
import loss_function
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

    frame_size = past_size + present_size + future_size

    @tf.function
    def test_step(dist_inputs):
        return_list = []
        def step_fn(inputs):
            index, x, _ = inputs
            if len(x.shape) == 0:
                x = tf.zeros([0, frame_size])
            y_pred = tf.squeeze(unet(tf.expand_dims(x, 2)), 2)

            local_bath_size = index.shape[0]
            if local_bath_size != 0:
                y_pred_list = tf.split(y_pred, num_or_size_splits=local_bath_size, axis=0)
                for i in range(local_bath_size):
                    return_list.append([index[i], tf.squeeze(y_pred_list[i])])

        multi_gpu_strategy.run(step_fn, args=(dist_inputs,))
        return return_list

    # load model
    if load_checkpoint_name != "":
        load_checkpoint_path = os.path.join('./checkpoint', load_checkpoint_name)
        unet.load_weights(os.path.join(load_checkpoint_path, 'data.ckpt'))
    else:
        util.raise_error("Check load_checkpoint_name.")

    # read file list
    source_file_list = util.read_path_list(test_source_path, 'wav')
    target_file_list = util.read_path_list(test_target_path, 'wav')

    # dataset initialize
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    for i, (source_file, target_file) in enumerate(zip(source_file_list, target_file_list)):
        test_dataset = dataset.AudioDataset(audio_file_path=[source_file, target_file],
                                             sampling_rate=sampling_rate, frame_size=frame_size, shift_size=shift_size,
                                             batch_size=batch_size, shuffle=False)
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
            temp_return_list = test_step(inputs)
            return_list.extend(temp_return_list)


        # output
        orig_output = np.zeros(test_dataset.get_total_length())
        for index, orig_pred in return_list:
            index = int(index)
            orig_pred = orig_pred.numpy()
            orig_output[index*shift_size+past_size:index*shift_size+past_size+frame_size] += orig_pred * util.custom_kbd(frame_size)
        orig_output = orig_output[test_dataset.get_padding_size()[0]:len(orig_output)-test_dataset.get_padding_size()[1]]
        util.write_audio_file(save_file_path, orig_output, sampling_rate)

        end_time = util.second_to_dhms_string(time.time()-start)
        print(f"({end_time})")
