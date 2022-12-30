import os
import util
import shutil
import pandas as pd


TimeUnet_path = "/home/ubuntu/SaturationEnhancement/subjective_file/TimeUnet"
Detection_path = "/home/ubuntu/SaturationEnhancement/subjective_file/Detection"

clip_csv_path = "/home/ubuntu/DATASET/VCTK_CLIP_dataset/clip.csv"
df = pd.read_csv(clip_csv_path)[['fileid', 'clip_gain']]

TimeUnet_file_list = util.read_path_list(TimeUnet_path, extension='wav')
print(TimeUnet_file_list)
Detection_file_list = util.read_path_list(Detection_path, extension='wav')
print(Detection_file_list)
with open('./subjective_file/AB5_CLIPPING.asi', "w") as f:
    f.write(f'session=AB5\n\n')
    for i, (file1, file2) in enumerate(zip(TimeUnet_file_list, Detection_file_list)):
        f.write(f'# item{i+1}\n')
        file1_path = os.path.join('./', os.path.basename(TimeUnet_path), os.path.basename(file1))
        file2_path = os.path.join('./', os.path.basename(Detection_path), os.path.basename(file2))

        # clip_gain = float(df[df.fileid == os.path.basename(file1)].clip_gain)
        # if clip_gain <= 0.5:
        #     f.write(f'{file1_path}\n')
        # else:
        #     f.write(f'{file2_path}\n')
        f.write(f'{file1_path}\n')
        f.write(f'{file2_path}\n\n')
