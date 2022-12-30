import os,  util, time
from multiprocessing import Pool
from pypesq import pesq
import numpy as np


if __name__ == '__main__':
    result_csv_file_path = "/home/ubuntu/SaturationEnhancement/Time_large/detection/frame_clipping_rate.csv"
    source_path = "/home/ubuntu/DATASET/VCTK_CLIP_dataset/"
    target_path = "/home/ubuntu/DATASET/VCTK_dataset/"
    sampling_rate = 16000
    frame_size = 1600
    shift_size = 800

    def find_frame_clipping_rate(source, target):
        total_frame_cnt = 0
        clipping_frame_cnt = 0
        if len(source) != len(target):
            print('error')

        for i in range(0, len(source), shift_size):
            x = source[i:i+frame_size]
            y = target[i:i+frame_size]

            label = np.where(np.sum(np.abs(y - x)) > 1e-5, 1., 0.)
            total_frame_cnt += 1
            clipping_frame_cnt += label

        frame_clipping_rate = clipping_frame_cnt / total_frame_cnt
        return frame_clipping_rate


    def func(source_file_path, target_file_path):
        start = time.time()

        if not os.path.isfile(source_file_path):
            print(f"fileid:{os.path.basename(source_path)} -> Not exist.")
            return None

        # read train data file
        source_signal = util.read_audio_file(source_file_path, sampling_rate)
        target_signal = util.read_audio_file(target_file_path, sampling_rate)

        out = find_frame_clipping_rate(source_signal, target_signal)

        print(f"fileid:{os.path.basename(source_file_path)} -> Success. ({util.second_to_dhms_string(time.time() - start, second_round=False)})")

        return [out]


    source_path_list = util.read_path_list(source_path, extension='wav')
    target_path_list = util.read_path_list(target_path, extension='wav')

    p = Pool(os.cpu_count())
    return_list = p.starmap(func, zip(source_path_list, target_path_list))
    p.close()
    p.join()

    if os.path.isdir((result_csv_file_path)):
        os.makedirs(os.path.dirname(result_csv_file_path), exist_ok=True)
    with open(result_csv_file_path, 'w') as f:
        f.write("fileid,frame_clipping_rate, use\n")
        for source_path, r in zip(source_path_list, return_list):
            tmp = source_path.split('/')
            if r:
                f.write(f"{tmp[-1]},{r[0]},{tmp[-3]}\n")

