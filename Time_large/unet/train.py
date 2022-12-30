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
    mse = loss_function.mse_loss_function()

    # model initialize
    unet = model.UNet(16)

    # dataset initialize
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    frame_size = past_size + present_size + future_size
    train_dataset = dataset.AudioDataset(audio_file_path=[train_source_path, train_target_path], sampling_rate=sampling_rate,
                                         frame_size=frame_size, shift_size=shift_size, batch_size=batch_size,
                                         shuffle=True, shuffle_seed=0, window_type='sine')
    dist_train_dataset = multi_gpu_strategy.experimental_distribute_dataset(train_dataset.get_dataset().with_options(options))

    if validation:
        valid_dataset = dataset.AudioDataset(audio_file_path=[valid_source_path, valid_target_path], sampling_rate=sampling_rate,
                                             frame_size=frame_size, shift_size=shift_size, batch_size=batch_size,
                                             shuffle=True, shuffle_seed=0, window_type='sine')
        dist_valid_dataset = multi_gpu_strategy.experimental_distribute_dataset(valid_dataset.get_dataset().with_options(options))


    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            idx, x, y_true = inputs
            if len(x.shape) == 0:
                x = tf.zeros([0, frame_size])
                y_true = tf.zeros([0, frame_size])

            with tf.GradientTape(persistent=False) as tape:
                y_pred = tf.squeeze(unet(tf.expand_dims(x, 2)), 2)
                loss = mse(tf.slice(y_true, [0, past_size], [-1, present_size]), tf.slice(y_pred, [0, past_size], [-1, present_size]))

            gradients = tape.gradient(loss, unet.trainable_variables)
            optimizer.apply_gradients(zip(gradients, unet.trainable_variables))
            return [loss]

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
            loss = mse(tf.slice(y_true, [0, past_size], [-1, present_size]), tf.slice(y_pred, [0, past_size], [-1, present_size]))
            return [loss]

        per_loss = multi_gpu_strategy.run(step_fn, args=(dist_inputs,))
        multi_gpu_loss = multi_gpu_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=None)
        return multi_gpu_loss


    # load model
    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        for inputs in dist_train_dataset:
            train_step(inputs)
            break
        load_checkpoint_path = os.path.join('./checkpoint', load_checkpoint_name)
        unet.load_weights(os.path.join(load_checkpoint_path, 'data.ckpt'))
        try:
            util.load_optimizer_state(optimizer, os.path.join(load_checkpoint_path, 'optimizer.npy'))
        except:
            util.raise_issue(f"Failed to load optimizer.")

        # learning_rate_decay
        optimizer.learning_rate = learning_rate * (0.99) ** (saved_epoch // 2)
    else:
        saved_epoch = 0

    # neptune loss init
    neptune.loss_init(['mse'], saved_epoch, category='train')
    neptune.loss_init(['mse'], saved_epoch, category='valid')

    # run
    for epoch in range(saved_epoch + 1, saved_epoch + epochs + 1):
        start = time.time()
        now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        train_mse_loss = 0
        num_of_iteration = math.ceil(train_dataset.get_number_of_frame() / batch_size)
        for iteration, inputs in enumerate(dist_train_dataset):
            print(f"\r({now_time})", end=" ")
            print(util.change_font_color('bright cyan', 'Train:'), end=" ")
            print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{saved_epoch + epochs},"), end=" ")
            print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{iteration + 1}/{num_of_iteration}"), end=" ")
            train_loss = train_step(inputs)
            train_mse_loss += train_loss[0]

        train_mse_loss /= num_of_iteration
        neptune.log('mse', train_mse_loss, epoch, category='train')

        if epoch % save_checkpoint_period == 0 or epoch == 1:
            save_checkpoint_path = f"./checkpoint/{save_checkpoint_name}_{epoch}"
            os.makedirs(save_checkpoint_path, exist_ok=True)
            unet.save_weights(os.path.join(save_checkpoint_path, 'data.ckpt'))
            util.save_optimizer_state(optimizer, os.path.join(save_checkpoint_path, 'optimizer.npy'))

        end_time = util.second_to_dhms_string(time.time() - start)
        print(util.change_font_color('bright black', '|'), end=" ")
        print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{train_mse_loss:.4E}"), end=" ")
        print(f"({end_time})")

        if validation:
            start = time.time()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            valid_mse_loss = 0
            num_of_iteration = math.ceil(valid_dataset.get_number_of_frame() / batch_size)
            for iteration, inputs in enumerate(dist_valid_dataset):
                print(f"\r({now_time})", end=" ")
                print(util.change_font_color('bright cyan', 'Valid:'), end=" ")
                print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{saved_epoch + epochs},"), end=" ")
                print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{iteration + 1}/{num_of_iteration}"), end=" ")
                valid_loss = valid_step(inputs)
                valid_mse_loss += valid_loss[0]

            valid_mse_loss /= num_of_iteration
            neptune.log('mse', valid_mse_loss, epoch, category='valid')

            end_time = util.second_to_dhms_string(time.time() - start)
            print(util.change_font_color('bright black', '|'), end=" ")
            print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{valid_mse_loss:.4E}"), end=" ")
            print(f"({end_time})")

        # learning_rate_decay
        optimizer.learning_rate = learning_rate * (0.99) ** (epoch // 2)

neptune.stop()