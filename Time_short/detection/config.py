##### LEARNING_PARAMETER #######################################################################
epochs                          = 100000
batch_size                      = 256
learning_rate                   = 0.001     # detection learning rate
##### CHECKPOINT_SETTING #######################################################################
unet_load_checkpoint_name       = "Saturation"
load_checkpoint_name            = ""
save_checkpoint_name            = "Detection"
save_checkpoint_period          = 1
validation                      = True
##### MODEL_PARAMETER ##########################################################################
past_size                       = 0
present_size                    = 320
future_size                     = 0
shift_size                      = 280
sampling_rate                   = 16000
##### DATASET_PATH #############################################################################
train_source_path               = "/home/ubuntu/DATASET/VCTK_CLIP_dataset/train"
test_source_path                = "/home/ubuntu/DATASET/VCTK_CLIP_dataset/test"
valid_source_path               = "/home/ubuntu/DATASET/VCTK_CLIP_dataset/valid"
train_target_path               = "/home/ubuntu/DATASET/VCTK_dataset/train"
test_target_path                = "/home/ubuntu/DATASET/VCTK_dataset/test"
valid_target_path               = "/home/ubuntu/DATASET/VCTK_dataset/valid"
##### NEPTUNE_SETTING ##########################################################################
project_name                    = 'csp-lab/MicEnhancement'
model_name                      = "DetectTS"
api_key_name                    = 'eunmi'
backup_file_list                = ['train.py', 'model.py', 'config.py', 'loss_function.py']
