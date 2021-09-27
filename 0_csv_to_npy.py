

import numpy as np
import os, cv2
csv_name='no_spectrogram_infoICA_56X56'
csv_dir= 'D:\\PycharmProjects\\EEG_signal_PSD_DE\\'+csv_name+'\\'
npy_dir='D:\\PycharmProjects\\signal_npy\\'+csv_name
npy_dir_no_data='D:\\PycharmProjects\\signal_npy\\'+csv_name+'(no_inf).npy'


def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False



def dataset2(csv_dir):
    dataset_X = []
    dataset_y = []
    filelist = os.listdir(csv_dir)
    filelist.sort()
    EEG_image=[]
    #for line in range(1000):
    for line in range(len(filelist)):
        if '.csv' in filelist[line]:
            img = np.loadtxt(csv_dir + filelist[line], delimiter=',')
        else:
            img = cv2.imread(csv_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        EEG_image.append(img)
        if line==len(filelist)-1 or filelist[line][:-8] != filelist[line + 1][:-8]:
            EEG_image= np.transpose(EEG_image, (1, 2, 0))
            if len(dataset_X)>1 and np.array(dataset_X[len(dataset_X)-1]).shape != np.array(EEG_image).shape:
                print(len(dataset_X)-1,'번 데이터가 다름:' ,np.array(EEG_image).shape, filelist[line])
            else:
                #print(np.array(dataset_X).shape)
                dataset_X.append(EEG_image)
                dataset_y.append(int(filelist[line][-5])-1)
                '''
                                if (int(filelist[line][-5]) > 3):
                    dataset_y.append(3)
                else:
                    dataset_y.append(int(filelist[line][-5]) - 1)
                '''

            EEG_image=[]

    dataset_X = np.array(dataset_X)
    number = dataset_X.shape[0]
    dataset_y=np.array(dataset_y)
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]
    dataset_inf = None
    return dataset_X, dataset_y, dataset_inf


def dataset2_plus(csv_dir):
    dataset_X,dataset_y,dataset_inf = [],[],[]

    filelist = os.listdir(csv_dir)
    filelist.sort()
    EEG_image=[]
    #for line in range(1000):
    for line in range(len(filelist)):
        if '.csv' in filelist[line]:
            img = np.loadtxt(csv_dir + filelist[line], delimiter=',')
        else:
            img = cv2.imread(csv_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        EEG_image.append(img)
        if line==len(filelist)-1 or filelist[line][:-8] != filelist[line + 1][:-8]:
            EEG_image= np.transpose(EEG_image, (1, 2, 0))
            if len(dataset_X)>1 and np.array(dataset_X[len(dataset_X)-1]).shape != np.array(EEG_image).shape:
                print(len(dataset_X)-1,'번 데이터가 다름:' ,np.array(EEG_image).shape, filelist[line])
            else:
                #print(np.array(dataset_X).shape)
                dataset_X.append(EEG_image)
                dataset_inf.append([filelist[line][5:8],filelist[line][18:22]])
                dataset_y.append(int(filelist[line][-5])-1)
                '''
                                if (int(filelist[line][-5]) > 3):
                    dataset_y.append(3)
                else:
                    dataset_y.append(int(filelist[line][-5]) - 1)
                '''

            EEG_image=[]

    dataset_X = np.array(dataset_X)
    number = dataset_X.shape[0]
    dataset_y=np.array(dataset_y)
    dataset_inf=np.array(dataset_inf)
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]
    dataset_inf= dataset_inf[s]

    return dataset_X, dataset_y, dataset_inf

if __name__ == "__main__":
    data_x, data_y, dat_inf= dataset2_plus(csv_dir)
    if (len(data_x) != len(data_y)):
        print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_y)))
    else:
        np.save( npy_dir_no_data+'_x.npy',np.array(data_x))
        np.save( npy_dir_no_data+'_y.npy',np.array(data_y))

        if dat_inf is not None:
            np.save( npy_dir+'.npy',np.array(dat_inf))




