


import numpy as np
import scipy.io as sio
import multiprocessing
import os
import csv




def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def start(dataDir, filelist):

    cut_line  = []
    eeg_lines_t = []

    with open(dataDir + '\\' + filelist + '\\bio\\' + filelist + '_mark.txt', 'r') as mat_data_mark:
        lines_mark = mat_data_mark.readlines()
        start= 0
        if len(lines_mark)>3:
            for line_start in lines_mark:
                if line_start.split()[0]=='RT:u':
                    break;
                else:
                    start+=1
            print(filelist ,start)
            [cut_line.append(float(lines_mark[line].split()[1])) for line in range(start-1, len(lines_mark)) if len(lines_mark[line]) > 4 and not (lines_mark[line-1].split()[0]=='RT:r' and lines_mark[line].split()[0]=='RT:r')]
    cut_line = [cut_line[line]+0.3 if line%2==0 else cut_line[line] for line in range(len(cut_line))] #이거는 opflow를 위해서 특별히 추가해서 r뒤에 +1 되는 문구입니다.
    score_learn=[]
    with open('D:\\2018임상데이터\\임상데이터_정리파일\\bio' + '\\' + filelist[0:16] + '\\bio\\' + filelist[0:16] + 'Score.csv', 'r',
                  encoding='utf-8') as f:
        score_line = csv.reader(f)
        [score_learn.append([str(sline[1]),str(sline[0])])for sline in score_line if isNumber(sline[0])]


    mat_data = open(dataDir + '\\' + filelist + '\\bio\\' + filelist + '.txt', 'r')
    lines = mat_data.readlines()
    i=0

    for lines in lines:
        lines=lines.split()
        if (len(cut_line)-1> 2*i and 52>i and len(score_learn)>i) and isNumber(lines[0]) and isNumber(lines[1]) :
            if cut_line[2*i] <=float(lines[0]) and cut_line[2*i+1]>float(lines[0]):
                eeg_lines_t.append([float(lines[n]) for n in range(1,len(lines))])
            elif cut_line[2*i+1]<float(lines[0]):
                #print('cut_line[', 2 * i, ']=', cut_line[2 * i], 'cut_line[', 2 * i + 1, ']=', cut_line[2 * i + 1])
                print('D:\\PycharmProjects\\EEG_signal_PSD_DE\\' + filelist + '_scene_' + score_learn[i][0],'_',score_learn[i][1] + '.csv')
                with open('D:\\PycharmProjects\\EEG_signal_PSD_DE\\data_cut_scene+0.3sec\\' + filelist +'_scene_'+score_learn[i][0]+'_'+score_learn[i][1]  +'.csv', 'w',  newline='', encoding = "utf-8") as save_flie:
                    wr = csv.writer(save_flie,  dialect='excel')
                    wr.writerows(np.array(eeg_lines_t).T)
                eeg_lines_t=[]
                i += 1


def mat_file_assert(dataDir):
    filelist_learning= os.listdir(dataDir)
    filelist_learning.sort()

    for filelist in filelist_learning:
        if os.path.isfile(dataDir + '\\' + filelist + '\\bio\\' + filelist + '_mark.txt') and os.path.isfile(dataDir + '\\' + filelist + '\\bio\\' + filelist + '.txt') :
        ##if os.path.isfile(dataDir + '\\' + filelist + '\\bio\\' + filelist + '_mark.txt') and os.path.isfile(dataDir + '\\' + filelist + '\\bio\\' + filelist + '.txt') :
            start(dataDir, filelist)


if __name__ == "__main__":

    mat_file_assert('D:\\2018임상데이터\\임상데이터_정리파일\\bio\\')
