


'''

CNN피쳐를 처음부터 끝까지 쭉 봅니다 진짜 쭉 모든걸 다

'''


from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2,os
from keras.models import load_model

filename = './models/model_ver_2019_09_25_19_26(epochs_100_num_classes_1).h5'
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
            layer_to= len(resnet18.layers)-2
            outputs = resnet18.layers[layer_to].output
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


    feature_maps_np= np.array(feature_maps)
    y_np= np.array(y)

def dataset():
    filters, biases = resnet18.layers[len(resnet18.layers)-7].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


if __name__ == "__main__":
    dataset()


