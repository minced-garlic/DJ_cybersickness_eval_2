


'''

이거 CNN 아니라 feature까서 넣는 거임......................................

'''


from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2,os
from keras.models import load_model

filename = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models\\model_ver_2019_10_08_11_47_best4_(epochs_10_num_classes_5).h5'
resnet18 = load_model(filename)
#dataset_x_dir='D:\\PycharmProjects\\EEG_signal_PSD_DE\\spectrogram_infoICA(w.rest)\\'
dataset_x_dir='D:\\2019_data_for_Cnn\\spectrogram_infoICA\\'
exportdir='C:/Users/Admin/PycharmProjects/EEG_signal_preprocessing_01/data/'
now = time.localtime()
file_naming= str(now.tm_year)+'_'+str(now.tm_mon)+'_'+str(now.tm_mday)+'_'+str(now.tm_hour)+'_'+str(now.tm_min)


def dataset2(dataset_x_dir):
    dataset_X = []
    y= []
    feature_maps= []
    filelist = os.listdir(dataset_x_dir)
    filelist.sort()
    for line in range(len(filelist)):
        if 'csv' in filelist[line]:
            img = np.loadtxt(dataset_x_dir+filelist[line], delimiter=',')
        else:
            img = cv2.imread(dataset_x_dir + filelist[line], cv2.IMREAD_GRAYSCALE)
        dataset_X.append(img)
        if line<len(filelist)-1 and filelist[line][:-8] != filelist[line + 1][:-8]:
            #print(len(resnet18.layers))
            outputs = resnet18.layers[len(resnet18.layers)-2].output
            model = Model(inputs=resnet18.inputs, outputs=outputs)
            img_name=filelist[line][:-5]+'.csv'
            img = np.array(dataset_X)
            img = img.reshape((56, 56, 8))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feature_map = model.predict(img)
            feature_maps.append(feature_map[0])
            y.append(int(filelist[line][-5]))
            dataset_X = []
            print(filelist[line][:-8],' completed')
        if len(y)%1000==0 and len(y)>0:
            feature_maps_np = np.array(feature_maps)
            y_np = np.array(y)
            np.savetxt(exportdir+'feature_' + file_naming + '_x.csv', feature_maps_np, delimiter=",")
            np.savetxt(exportdir+'feature_' + file_naming + '_y.csv', y_np, delimiter=",")
            feature_maps_np, y_np= [], []

    feature_maps_np= np.array(feature_maps)
    y_np= np.array(y)
    if(len(y)!=len(feature_maps_np)):
        print("길이가 같지 않습니다: len(feature_maps_np)", len(feature_maps_np), "len(y)",len(y) )
    else:
        np.savetxt(exportdir+'feature_'+file_naming+'_x.csv',feature_maps_np, delimiter="," )
        np.savetxt(exportdir+'feature_'+file_naming+'_y.csv',y_np, delimiter="," , fmt='%i')

if __name__ == "__main__":
    dataset2(dataset_x_dir)


