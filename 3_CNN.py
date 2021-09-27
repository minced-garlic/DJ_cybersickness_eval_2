'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

from __future__ import print_function
import numpy as np
import os, cv2, time
from attention_model.attention_module import attach_attention_module
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model
#dataset_dir = 'D:/PycharmProjects/EEG_signal_PSD_DE/'
dataset_dir = 'D:/2019_data_for_Cnn/'
dataset_x_dir= 'spectrogram_infoICA/'

# Training parameters

y_dir ='C:\\Users\\Admin\\PycharmProjects\\EEG_signal_PSD_DE/Score_mean.csv'
train = 4
test = 1
validation_split= 0.3
batch_size = 128
epochs = 5
data_augmentation = False
num_classes = 1
depth = 14
base_model = 'resnet'+str(depth)
attention_module = 'cbam_block'
model_type = base_model if attention_module==None else base_model+'_'+attention_module

def dataset2(dataset_x_dir, train, test):
    dataset_X = []
    dataset_y = []
    filelist = os.listdir(dataset_dir+dataset_x_dir)
    filelist.sort()
    EEG_image=[]
    for line in range(len(filelist)):
        img = cv2.imread(dataset_dir+dataset_x_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        #img =cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        EEG_image.append(img)
        if line==len(filelist)-1 or filelist[line][:-8] != filelist[line + 1][:-8]:
            EEG_image= np.transpose(EEG_image, (1, 2, 0))
            if len(dataset_X)>1 and np.array(dataset_X[len(dataset_X)-1]).shape != np.array(EEG_image).shape:
                print(len(dataset_X)-1,'번 데이터가 다름:' ,np.array(EEG_image).shape, filelist[line])
            else:
                #print(np.array(dataset_X).shape)
                dataset_X.append(EEG_image)
                dataset_y.append(int(filelist[line][-5]))
            EEG_image=[]

    dataset_X = np.array(dataset_X)
    number = dataset_X.shape[0]
    dataset_y=np.array(dataset_y)
    dataset_y= (dataset_y.astype('float32') -1)/ 4

    return dataset_X[int(number*test/(train+test)):], dataset_y[int(number*test/(train+test)):], dataset_X[:int(number*test/(train+test))], dataset_y[:int(number*test/(train+test))]

# 데이터 셋 2와 다른 점은 여긴 1,1,1,1,1로 5개 매핑임
def dataset3(dataset_x_dir, train, test):
    dataset_X = []
    dataset_y = []
    filelist = os.listdir(dataset_dir+dataset_x_dir)
    filelist.sort()
    EEG_image=[]
    for line in range(len(filelist)):
        img = cv2.imread(dataset_dir+dataset_x_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        #img =cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        EEG_image.append(img)
        if line==len(filelist)-1 or filelist[line][:-8] != filelist[line + 1][:-8]:
            EEG_image= np.transpose(EEG_image, (1, 2, 0))
            if len(dataset_X)>1 and np.array(dataset_X[len(dataset_X)-1]).shape != np.array(EEG_image).shape:
                print(len(dataset_X)-1,'번 데이터가 다름:' ,np.array(EEG_image).shape, filelist[line])
            else:
                #print(np.array(dataset_X).shape)
                dataset_X.append(EEG_image)
                y= [0,0,0,0,0]
                y[int(filelist[line][-5])-1]=1
                dataset_y.append(y)
            EEG_image=[]

    dataset_X = np.array(dataset_X)
    number = dataset_X.shape[0]
    dataset_y=np.array(dataset_y)
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]


    return dataset_X[int(number*test/(train+test)):], dataset_y[int(number*test/(train+test)):], dataset_X[:int(number*test/(train+test))], dataset_y[:int(number*test/(train+test))]


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes= num_classes, attention_module=None):

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(2):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            # attention_module
            if attention_module is not None:
                y = attach_attention_module(y, attention_module)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Resnet(비슷한 거)
def image_CNN(x_train, y_train, x_test, y_test):
        input_shape = x_train.shape[1:]

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        model = resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)


        time_start= time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))
        from keras.models import load_model

        if num_classes== 1:
            model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
            model.summary()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,shuffle=True)
            filename = './models/model_ver'+time_start+'(epochs_'+str(epochs)+'_num_classes_'+str(num_classes)+').h5'
            model.save(filename)
            load_model = load_model(filename)
            pred_X= np.array(load_model.predict(x_test)).T[0]
            pred_train_X = np.array(load_model.predict(x_train)).T[0]
            predscore=np.vstack((pred_X,y_test))
            predtrain= np.vstack((pred_train_X,y_train))
            np.savetxt('./predicts/predict_test_'+dataset_x_dir[:-3]+time_start+'(epochs_'+str(epochs)+'_SVR_'+str(num_classes)+').csv',(predscore*4+1).T, fmt= '%.3f', delimiter=",")
            np.savetxt('./predicts/predict_score_'+dataset_x_dir[:-3]+time_start+'(epochs_'+str(epochs)+'_SVR_'+str(num_classes)+').csv',(predtrain*4+1).T, fmt= '%.3f', delimiter=",")
        elif num_classes>1:
            from sklearn import  metrics
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,shuffle=True, verbose=0)
            filename = './models/model_ver_'+time_start+'(epochs_'+str(epochs)+'_num_classes_'+str(num_classes)+').h5'
            model.save(filename)
            load_model = load_model(filename)
            pred_X=  np.array(load_model.predict(x_test).argmax(axis=-1)).T
            pred_train_X =  np.array(load_model.predict(x_train).argmax(axis=-1)).T
            y_test=y_test.argmax(axis=-1).T
            y_train=y_train.argmax(axis=-1).T
            np.savetxt('./predicts/confusionmat_test'+time_start+'(epochs_'+str(epochs)+').csv', metrics.confusion_matrix(pred_X, y_test),
                       delimiter=',')
            np.savetxt('./predicts/confusionmat_train'+time_start+'(epochs_'+str(epochs)+').csv', metrics.confusion_matrix(pred_train_X,y_train),
                       delimiter=',')
        else:
            raise Exception("'{}' is not supported attention module!".format(num_classes))

if __name__ == "__main__":
    if num_classes==1:
        data_train_x, data_train_y, data_x, data_y= dataset2(dataset_x_dir, train, test)
    elif num_classes ==5:
        data_train_x, data_train_y, data_x, data_y= dataset3(dataset_x_dir, train, test)
    if (len(data_x) != len(data_y)):
        print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_y)))
    else:
        image_CNN(data_train_x, data_train_y,data_x,data_y)


