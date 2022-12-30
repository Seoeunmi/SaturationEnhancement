##### LEARNING_PARAMETER #######################################################################
batch_size                      = 256
##### CHECKPOINT_SETTING #######################################################################
unet_load_checkpoint_name       = "Saturation_"
load_checkpoint_name            = "Detection_"
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
