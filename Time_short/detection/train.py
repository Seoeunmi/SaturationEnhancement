import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import time
import datetime
import math
import tensorflow as tf
import neptune_function
from api_key import *
from config import *
import util
import model
import dataset
import loss_function
import random
import numpy as np

# neptune init
neptune = neptune_function.Neptune(project_name=project_name, model_name=model_name, api_key=api_key[api_key_name], file_names=backup_file_list)
neptune.save_parameter({'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate, 'past_size': past_size, 'present_size': present_size, 'future_size': future_size, 'shift_size': shift_size})

# set seed
seed = int(42)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)

# multi gpu scope
multi_gpu_strategy = tf.distribute.MirroredStrategy()
with multi_gpu_strategy.scope():
    # optimizer setting
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bce = loss_function.binary_cross_entropy_function()

    # model initialize
    unet = model.UNet(16)
    detection = model.Detection()

    # dataset initialize
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    frame_size = past_size + present_size + future_size
    train_dataset = dataset.AudioDataset(audio_file_path=[train_source_path, train_target_path], sampling_rate=sampling_rate,
                                         frame_size=frame_size, shift_size=shift_size, batch_size=batch_size,
                                         shuffle=True, shuffle_seed=0)
    dist_train_dataset = multi_gpu_strategy.experimental_distribute_dataset(train_dataset.get_dataset().with_options(options))

    if validation:
        valid_dataset = dataset.AudioDataset(audio_file_path=[valid_source_path, valid_target_path], sampling_rate=sampling_rate,
                                             frame_size=frame_size, shift_size=shift_size, batch_size=batch_size,
                                             shuffle=True, shuffle_seed=0)
        dist_valid_dataset = multi_gpu_strategy.experimental_distribute_dataset(valid_dataset.get_dataset().with_options(options))


    @tf.function
    def train_step(dist_inputs):
        """
            x : clipping
            y_true : orig wav
            y_pred : model restoration
        """
        def step_fn(inputs):
            idx, x, y_true = inputs
            if len(x.shape) == 0:
                x = tf.zeros([0, frame_size])
                y_true = tf.zeros([0, frame_size])

            y_pred = tf.squeeze(unet(tf.expand_dims(x, 2)), 2)
            labels = tf.where(tf.reduce_sum(tf.abs(y_true - x), axis=1) > 1e-5, 1., 0.)
            with tf.GradientTape(persistent=False) as tape:
                predictions = tf.squeeze(detection(tf.expand_dims(x, 2), tf.expand_dims(y_pred, 2)), axis=1)
                loss = bce(labels, predictions)

            gradients = tape.gradient(loss, detection.trainable_variables)
            optimizer.apply_gradients(zip(gradients, detection.trainable_variables))

            wrong_cnt = tf.reduce_sum(tf.abs(labels - tf.round(predictions)))
            correct_cnt = x.shape[0] - wrong_cnt
            return [loss, correct_cnt, wrong_cnt]

        per_loss = multi_gpu_strategy.run(step_fn, args=(dist_inputs,))
        multi_gpu_loss = multi_gpu_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=None)
        return multi_gpu_loss


    @tf.function
    def valid_step(dist_inputs):
        def step_fn(inputs):
            idx, x, y_true = inputs
            if len(x.shape) == 0:
                x = tf.zeros([0, frame_size])
                y_true = tf.zeros([0, frame_size])

            y_pred = tf.squeeze(unet(tf.expand_dims(x, 2)), 2)
            labels = tf.where(tf.reduce_sum(tf.abs(y_true - x), axis=1) > 1e-5, 1., 0.)
            predictions = tf.squeeze(detection(tf.expand_dims(x, 2), tf.expand_dims(y_pred, 2)), axis=1)
            loss = bce(labels, predictions)
            wrong_cnt = tf.reduce_sum(tf.abs(labels - tf.round(predictions)))
            correct_cnt = x.shape[0] - wrong_cnt
            return [loss, correct_cnt, wrong_cnt]

        per_loss = multi_gpu_strategy.run(step_fn, args=(dist_inputs,))
        multi_gpu_loss = multi_gpu_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=None)
        return multi_gpu_loss


    # load model
    if unet_load_checkpoint_name != "":
        unet_load_checkpoint_path = os.path.join('./checkpoint', unet_load_checkpoint_name)
        unet.load_weights(os.path.join(unet_load_checkpoint_path, 'data.ckpt'))

    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        for inputs in dist_train_dataset:
            train_step(inputs)
            break

        load_checkpoint_path = os.path.join('./checkpoint', load_checkpoint_name)
        detection.load_weights(os.path.join(load_checkpoint_path, 'detection_data.ckpt'))
        try:
            util.load_optimizer_state(optimizer, os.path.join(load_checkpoint_path, 'optimizer.npy'))
        except:
            util.raise_issue(f"Failed to load optimizer.")
    else:
        saved_epoch = 0

    # neptune loss init
    neptune.loss_init(['bce'], saved_epoch, category='train')
    neptune.loss_init(['accuracy'], saved_epoch, category='train')
    neptune.loss_init(['bce'], saved_epoch, category='valid')
    neptune.loss_init(['accuracy'], saved_epoch, category='valid')

    # run
    for epoch in range(saved_epoch + 1, saved_epoch + epochs + 1):
        start = time.time()
        now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        train_bce_loss = 0
        correct_cnt, wrong_cnt = 0, 0
        num_of_iteration = math.ceil(train_dataset.get_number_of_frame() / batch_size)
        for iteration, inputs in enumerate(dist_train_dataset):
            print(f"\r({now_time})", end=" ")
            print(util.change_font_color('bright cyan', 'Train:'), end=" ")
            print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{saved_epoch + epochs},"), end=" ")
            print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{iteration + 1}/{num_of_iteration}"), end=" ")
            train_loss = train_step(inputs)
            train_bce_loss += train_loss[0]
            correct_cnt += train_loss[1]
            wrong_cnt += train_loss[2]

        train_bce_loss /= num_of_iteration
        train_accuracy = correct_cnt / (correct_cnt + wrong_cnt)
        neptune.log('bce', train_bce_loss, epoch, category='train')
        neptune.log('accuracy', train_accuracy, epoch, category='train')

        if epoch % save_checkpoint_period == 0 or epoch == 1:
            save_checkpoint_path = f"./checkpoint/{save_checkpoint_name}_{epoch}"
            os.makedirs(save_checkpoint_path, exist_ok=True)
            detection.save_weights(os.path.join(save_checkpoint_path, 'detection_data.ckpt'))
            util.save_optimizer_state(optimizer, os.path.join(save_checkpoint_path, 'optimizer.npy'))

        end_time = util.second_to_dhms_string(time.time() - start)
        print(util.change_font_color('bright black', '|'), end=" ")
        print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{train_bce_loss:.4E}"), end=" ")
        print(f"({end_time})")

        if validation:
            start = time.time()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            valid_bce_loss = 0
            correct_cnt, wrong_cnt = 0, 0
            num_of_iteration = math.ceil(valid_dataset.get_number_of_frame() / batch_size)
            for iteration, inputs in enumerate(dist_valid_dataset):
                print(f"\r({now_time})", end=" ")
                print(util.change_font_color('bright cyan', 'Valid:'), end=" ")
                print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{saved_epoch + epochs},"), end=" ")
                print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{iteration + 1}/{num_of_iteration}"), end=" ")
                valid_loss = valid_step(inputs)
                valid_bce_loss += valid_loss[0]
                correct_cnt += valid_loss[1]
                wrong_cnt += valid_loss[2]

            valid_bce_loss /= num_of_iteration
            valid_accuracy = correct_cnt / (correct_cnt + wrong_cnt)
            neptune.log('bce', valid_bce_loss, epoch, category='valid')
            neptune.log('accuracy', valid_accuracy, epoch, category='valid')

            end_time = util.second_to_dhms_string(time.time() - start)
            print(util.change_font_color('bright black', '|'), end=" ")
            print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{valid_bce_loss:.4E}"), end=" ")
            print(f"({end_time})")

neptune.stop()