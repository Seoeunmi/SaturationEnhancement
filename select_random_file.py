import os
import util
import shutil
import pandas as pd
import random

random.seed(10)

input_path = "/home/ubuntu/SaturationEnhancement/Time_large/unet/test_result/Saturation_200"
output_path = "/home/ubuntu/SaturationEnhancement/subjective_file/TimeUnet"
clip_csv_path = "/home/ubuntu/DATASET/VCTK_CLIP_dataset/clip.csv"


if not os.path.isfile(clip_csv_path):
    util.raise_error('meta.csv file is not exist in aec_synth_dataset_path.')
df = pd.read_csv(clip_csv_path)[['fileid', 'clip_gain']]
clip_file_list = [[] for _ in range(10)]

input_file_list = util.read_path_list(input_path, extension='wav')
for path in input_file_list:
    clip_gain = float(df[df.fileid == os.path.basename(path)].clip_gain)
    clip_file_list[int(clip_gain * 10)].append(path)


for i in range(1, 10):
    random_file_list = random.sample(clip_file_list[i], 15)
    for file_path in random_file_list:
        save_path = os.path.join(output_path, os.path.basename(file_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy(file_path, save_path)


#
# os.makedirs(output_path, exist_ok=True)
# with open('./subjective_file/subjective_clip.csv', "w") as f:
#     for i in range(1, 10):
#         random_file_list = random.sample(clip_file_list[i], 15)
#         for file_path in random_file_list:
#             save_path = os.path.join(output_path, os.path.basename(file_path))
#             shutil.copy(file_path, save_path)
#
#             f.write(f'{os.path.basename(file_path)},{i/10}\n')