import os
import numpy as np
save_dir='D:/2019_data_for_Cnn/rest_raw\\'
dataset_dir='D:\\2019임상데이터\\원본\\experimental_data_bio\\'


def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

filelist = os.listdir(dataset_dir)
filelist.sort()
for file in filelist:
    if 'rest.txt' in file:
            mat_data = open(dataset_dir + file, 'r')
            lines = mat_data.readlines()
            eeg_lines_t = []
            for lines in lines:
                lines = lines.split()
                if len(lines)>7 and isNumber(lines[0]):
                    #print(lines)
                    eeg_lines_t.append([float(lines[n]) for n in range(1, len(lines))])
            x = np.array(eeg_lines_t).T
            print(save_dir + file[:-4]  +'.npy')
            np.save(save_dir + file[:-4]  +'.npy',x)
