
import os, shutil
import numpy as np

dir_name='C:/Users/Admin/PycharmProjects/DJ_cybersickness_eval_2/predicts/subjects/'
save_name='C:/Users/Admin/PycharmProjects/DJ_cybersickness_eval_2/predicts/all_subjects_until_'


if __name__ == "__main__":
    filelist_learning = os.listdir(dir_name)
    filelist_learning.sort()
    all_average=[[' '],[1],[2],[3],[4],[5],['error1'],['error2'],['error3'],['error4'],['error5']]
    score= [[], [], [], [], []] # 0~4
    for filelist in filelist_learning:
        all_average[0].append(filelist[-11:-4])
        predict_list = np.loadtxt(dir_name+filelist, delimiter=",")
        [score[int(predict[1])-1].append(predict[0]) for predict in predict_list]
        [all_average[i+1].append(round(np.average(score[i]),3)) if len(score[i])!= 0 else all_average[i+1].append('') for i in range(5)]
        [all_average[i+6].append(round(np.std(score[i])/np.sqrt(len(score[i])),3)) if len(score[i])!= 0 else all_average[i+6].append('') for i in range(5)]
        score = [[], [], [], [], []]  # 0~4
    np.savetxt(save_name+filelist[-11:], np.array(all_average).T,fmt="%s", delimiter=",")




