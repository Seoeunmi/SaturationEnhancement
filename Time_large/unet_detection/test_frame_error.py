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
            out = predictions * y_pred + (1 - predictions) * x
            clipping_region = tf.where(tf.abs(y_true - x) > 1e-5, 1., 0.)
            labels = tf.where(tf.reduce_sum(tf.abs(y_true - x), axis=1) > 1e-5, 1., 0.)


            local_bath_size = index.shape[0]
            if local_bath_size != 0:
                out_list = tf.split(out, num_or_size_splits=local_bath_size, axis=0)
                x_list = tf.split(x, num_or_size_splits=local_bath_size, axis=0)
                predictions_list = tf.split(predictions, num_or_size_splits=local_bath_size, axis=0)
                labels_list = tf.split(labels, num_or_size_splits=local_bath_size, axis=0)
                y_true_list = tf.split(y_true, num_or_size_splits=local_bath_size, axis=0)
                clipping_region_list = tf.split(clipping_region, num_or_size_splits=local_bath_size, axis=0)
                for i in range(local_bath_size):
                    return_list.append([index[i], tf.squeeze(out_list[i]), tf.squeeze(x_list[i]), predictions_list[i], labels_list[i], tf.squeeze(y_true_list[i]), tf.squeeze(clipping_region_list[i])])

        multi_gpu_strategy.run(step_fn, args=(dist_inputs,))
        return return_list


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
    with open('./frame_error.csv', "w") as f:
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
                temp_return_list = test_step(inputs)
                return_list.extend(temp_return_list)
            # output
            for index, declipped_signal, clipped_signal, prediction, label, y_true, clipping_region in return_list:
                if label == 1 and prediction == 0:
                    for true_sample in y_true:
                        f.write(f"{true_sample},")
                    f.write("\n")
                    for orig_pred_sample in clipped_signal:
                        f.write(f"{orig_pred_sample},")
                    f.write("\n")
                    for region in clipping_region:
                        f.write(f"{region},")
                    f.write("\n\n")

                    # index = int(index)
                    # orig = orig.numpy()
                    # orig_pred = orig_pred.numpy()
                    # y_true = y_true.numpy()
                    #
                    # plt.plot(clipping_region * max(y_true), color='green')
                    # plt.plot(y_true)
                    # # plt.plot(orig_pred)
                    # plt.plot(orig)
                    #
                    # fig_path = f'./plot_error_frame/orig_clip/{save_file_name.replace(".wav", f"_{index}.png")}'
                    # os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                    # plt.savefig(fig_path)
                    # # plt.show()
                    # plt.close()

            end_time = util.second_to_dhms_string(time.time()-start)
            print(f"({end_time})")
