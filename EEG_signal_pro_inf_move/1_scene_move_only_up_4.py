from scipy import stats
import numpy as np
import os,time

pre_dir= 'D:/PycharmProjects/EEG_signal_PSD_DE/data_cut_scene_infoICA/'
save_dir= 'D:/PycharmProjects/EEG_signal_PSD_DE/pvalue/'


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
    '''
    return np.array([np.average(mat_i[delta:theta]), np.average(mat_i[theta:alpha]),
                     np.average(mat_i[alpha:beta]), np.average(mat_i[beta:gamma]),
                     np.average(mat_i[gamma:end])]).tolist()
    
    '''
    return np.average(mat_i[theta:alpha])

def moving_ave(mat, term):
    ma_mat = np.array(mat)
    for i in range(term, len(mat) - term):
        ma_mat[i] = np.mean(mat[i - term:i + term])
    return ma_mat

def bandwave_calculate(restfileline):
    restfileline= np.array(restfileline)
    mat, freq = STFT(restfileline)
    aver_bandwave=wave_sepa(mat, freq )
    return aver_bandwave


def score_in():
    filelist_learning = os.listdir(pre_dir)
    filelist_learning.sort()
    score= [[],[],[],[],[]] # 0~4
    for num in range(len(filelist_learning)):
        data= np.loadtxt(pre_dir+filelist_learning[num] , delimiter=",").tolist()
        score_num=int(filelist_learning[num][-5])-1
        score[score_num]=[bandwave_calculate(dataline[-256*(i+1)-1:-256*i-1]) for dataline in data for i in range(8)]
        if filelist_learning[num][:8]!=filelist_learning[num-1][:8]:
            score= [(np.array(scoreN).reshape(-1)).tolist() for scoreN in score]
            if len(score[1])+len(score[2])+len(score[3])+len(score[4])!=0:
            #if len(score[2]) != 0 and len(score[3]) + len(score[4]) != 0:
                to_print= np.array([stats.ks_2samp(score[0], score[i]) for i in range(1, len(score))  if len(score[i])!=0 ]).T
                print(filelist_learning[num - 1][:7], ' ', to_print[1] < 0.05)
            else:
                print(filelist_learning[num - 1][:7], ' False' )
            score = [[], [], [], [], []]


def score_out():
    filelist_learning = os.listdir(pre_dir)
    filelist_learning.sort()
    score= [] # number
    score_name=[]
    for bandwave in range(5):
        for num in range(int(len(filelist_learning))):
            score_num=int(filelist_learning[num][-5])-1
            #print(score_num)
            if score_num ==0:
                data= np.loadtxt(pre_dir+filelist_learning[num] , delimiter=",").tolist()
                score.append([bandwave_calculate(dataline[-256*(i+1)-1:-256*i-1])[bandwave] for dataline in data for i in range(8)])
                score_name.append(filelist_learning[num])
                #print(filelist_learning[num], ' appended')
        Allscore= (np.array(score).reshape(-1)).tolist()
        score= [(np.array(score_i).reshape(-1)).tolist() for score_i in score]
        [print(score_name[i][:-6], ' ', stats.ks_2samp(Allscore, score[i]),stats.ks_2samp(Allscore, score[i])[1]>0.01) for i in range(len(score))if len(score[i]) != 0]
        to_print = []
    '''
    
        for i in range(len(score)):
        if len(score[i]) != 0:
            to_print.append(stats.ks_2samp(Allscore, score[i])[1])
        if len(score)==i or score_name[i][:-6] != score_name[i+1][:-6]:
            print(score_name[i][:8], ' ', to_print < 0.05)
            to_print = []
            
    '''



def score_out2():
        filelist_learning = os.listdir(pre_dir)
        filelist_learning.sort()
        score= [[]] # number
        score_name=[]
        Allscore= []
        score_i=0
        for num in range(int(len(filelist_learning))):
            score_num=int(filelist_learning[num][-5])-1
            if score_num ==1:
                data= np.loadtxt(pre_dir+filelist_learning[num] , delimiter=",").tolist()
                app_end= [bandwave_calculate(dataline[-256*(i+1)-1:-256*i-1]) for dataline in data for i in range(8)]
                for x in app_end:
                    score[score_i].append(x)
                    Allscore.append(x)

                print(filelist_learning[num], ' appended')
            if int(len(filelist_learning)-1)>num and filelist_learning[num][:10] != filelist_learning[num+1][:10]:
                score.append([])
                score_name.append(filelist_learning[num][:7])
                score_i += 1

        score_name.append(filelist_learning[num][:7])
        to_print = []
        for i in range(len(score)):
            if len(score[i]) != 0:
                score[i]= (np.array(score[i]).reshape(-1)).tolist()
                aaaa=stats.ks_2samp(Allscore, score[i])
                to_print.append([score_name[i],aaaa[0],aaaa[1]])
                print(score_name[i],aaaa[0],aaaa[1])
        np.savetxt(save_dir+'p-value'+time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))+'.csv', to_print,fmt='%s', delimiter=",")
# aaaa[0]>> KS 통계량 :: K-S 통계량이 20 이상인 경우 모형의 변별력이 확보되는 것으로 판단
# aaaa[1]>> 유의확률 :: 0.05 양방향


if __name__ == "__main__":
    score_in()

