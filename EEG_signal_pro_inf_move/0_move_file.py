
import os, shutil
import numpy as np
false_file_dir ='D:\\PycharmProjects\\EEG_signal_PSD_DE\\cnn_01.txt'
pre_dir= 'D:\\PycharmProjects\\EEG_signal_PSD_DE\\spectrogram_infoICA/'
post_dir= 'D:\\PycharmProjects\\EEG_signal_PSD_DE\\spectrogram_infoICA(CNN2)/'

if __name__ == "__main__":
    false_file= np.loadtxt(false_file_dir,dtype='str', delimiter=",")
    filelist_learning = os.listdir(pre_dir)
    filelist_learning.sort()

    for filelist in filelist_learning:
        error = 0
        for false_list in false_file:
            if false_list in filelist:
                error = 1
        if error==0:
            shutil.copy(pre_dir+filelist,post_dir+filelist)
            print(pre_dir+filelist,'>>>>',post_dir+filelist)
