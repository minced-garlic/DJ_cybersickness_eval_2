

import numpy as np
import os, csv

datadir='D:\\PycharmProjects\\EEG_signal_PSD_DE\\'
def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def STFT( mat):
    fs = 256
    freq = np.fft.fftfreq(len(mat), 1/fs)
    return np.power(np.fft.fft(mat).real, 2)/ fs , freq

def wave_sepa(mat_i, freq):
    delta = 0
    theta = 0
    alpha = 0
    beta = 0
    gamma = 0
    end = 0
    for i in range(int(len(freq) / 2)):
        if np.abs(freq[i]) < 1:
            delta = int(i + 1)
        elif freq[i] < 4:
            theta = int(i + 1)
        elif freq[i] < 8:
            alpha = int(i + 1)
        elif freq[i] < 14:
            beta = int(i + 1)
        elif freq[i] < 31:
            gamma = int(i + 1)
        elif freq[i] < 50:
            end = int(i + 1)

    # 리턴값 : 세타, 등 전체 리턴값.
    return np.array([np.average(mat_i[delta:theta]), np.average(mat_i[theta:alpha]),
                     np.average(mat_i[alpha:beta]), np.average(mat_i[beta:gamma]),
                     np.average(mat_i[gamma:end])]).tolist()


def moving_ave(mat, term):
    ma_mat = np.array(mat)
    for i in range(term, len(mat) - term):
        ma_mat[i] = np.mean(mat[i - term:i + term])
    return ma_mat

def bandwave_calculate(restfileline):
    aver_bandwave=[]
    restfileline= restfileline.T
    for line in range(len(restfileline)):
        restfileline[line] = moving_ave(restfileline[line],5)
        mat, freq = STFT(restfileline[line])
        aver_bandwave.append(wave_sepa(mat, freq ))
    return aver_bandwave

def rest_file_assert(dataDir):
    filelist_learning= os.listdir(dataDir)
    filelist_learning.sort()
    restfile_overall=[]
    restfile_mid=[]
    for filelist in filelist_learning:
        restfiledir= dataDir + filelist + '\\rest\\' + filelist + '_rest.txt'
        rest_all=[]
        if os.path.isfile(restfiledir):
            with open(restfiledir, 'r') as mat_data_mark:
                rest_raw = mat_data_mark.readlines()
                for restallline in rest_raw:
                    if len(restallline) > 33 and isNumber(restallline[:8]):
                        rest_all.append([float(floatline) for floatline in restallline[:-2].split('\t')])
            rest_all= np.array(rest_all)[:][1:]
            lenw= len(rest_all)+3
            restfile_overall=bandwave_calculate(rest_all[256+3:-256])
            restfile_mid=bandwave_calculate(rest_all[int(lenw/2)-2560:int(lenw/2)+2560])

            np.savetxt(datadir+'/rest_overall_ICA/'+filelist+'_overall.csv', restfile_overall, delimiter=",")
            np.savetxt(datadir+'/rest_mid_ICA/'+filelist+'_mid.csv', restfile_mid, delimiter=",")

def rest_file_assert_2(dataDir):
    filelist_learning= os.listdir(dataDir)
    filelist_learning.sort()
    restfile_mid=[]
    for filelist in filelist_learning:
        restfiledir= dataDir + filelist + '\\rest\\' + filelist + '_rest.txt'
        rest_all=[]
        if os.path.isfile(restfiledir):
            with open(restfiledir, 'r') as mat_data_mark:
                rest_raw = mat_data_mark.readlines()
                for restallline in rest_raw:
                    if len(restallline) > 33 and isNumber(restallline[:8]):
                        rest_all.append([float(floatline) for floatline in restallline[:-2].split('\t')])
            rest_all= np.array(rest_all)[:][1:]
            lenw= len(rest_all)+3
            rest_all=rest_all[int(lenw/2)-2560:int(lenw/2)+2560].T
            np.savetxt('./rest_raw/'+filelist+'_raw.csv', rest_all, delimiter=",")


def rest_file_assert3(dataDir):
    filelist_learning= os.listdir(dataDir)
    filelist_learning.sort()
    restfile_overall=[]
    restfile_mid=[]
    for filelist in filelist_learning:
        if os.path.isfile(dataDir+filelist):
            rest_all= np.loadtxt(dataDir+filelist, delimiter=",").T
            lenw= len(rest_all)+3
            restfile_overall=bandwave_calculate(rest_all[256+3:-256])
            restfile_mid=bandwave_calculate(rest_all[int(lenw/2)-2560:int(lenw/2)+2560])

            np.savetxt(datadir+'rest_overall_infoICA/'+filelist[:16]+'_overall_infoICA.csv', restfile_overall, delimiter=",")
            np.savetxt(datadir+'rest_mid_infoICA/'+filelist[:16]+'_mid_infoICA.csv', restfile_mid, delimiter=",")


if __name__ == "__main__":
    #rest_file_assert('D:\\2018임상데이터\\임상데이터_정리파일\\rest\\')

    rest_file_assert3(datadir+'rest_infoICA/')
