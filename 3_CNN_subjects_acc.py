'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

import numpy as np
import os, cv2, time
import matplotlib.pyplot as plt


dataset_x_dir = 'D:/PycharmProjects/EEG_signal_PSD_DE/spectrogram_infoICA\\'

shape_cnn =(56,56,8)
model_is_rm = 1 # 1이면 회귀, 0이면 분류


def dataset(dataset_x_dir, test_sub):
    dataset_X = [[]]
    test_X=[[]]
    test_y = []
    dataset_y = []
    filelist = os.listdir(dataset_x_dir)
    filelist.sort()
    i=0
    ti=0
    for line in range(len(filelist)):
        img = cv2.imread(dataset_x_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        if test_sub in filelist[line]:
            test_X[ti].append(img)
            if line < len(filelist) - 1 and filelist[line][:-8] != filelist[line + 1][:-8]:
                test_X.append([])
                ti += 1
                test_y.append((int(filelist[line][-5]) - 1) / 4)
                if np.array( test_X[ti-1]).shape != np.array( test_X[ti-2]).shape and ti>1:
                    print(i,'번 데이터가 다름:' ,np.array(test_X[ti]).shape)
        else:
            dataset_X[i].append(img)
            if line<len(filelist)-1 and filelist[line][:-8] != filelist[line + 1][:-8]:
                dataset_X.append([])
                '''
                if int(filelist[line][-5])>3:
                    dataset_y.append(1)
                else:
                    dataset_y.append((int(filelist[line][-5])-1)/2)            
                '''
                dataset_y.append((int(filelist[line][-5])-1)/4)
                if np.array(dataset_X[i]).shape != np.array(dataset_X[i-1]).shape and i>1:
                    print(i,'번 데이터가 다름:' ,np.array(dataset_X[i]).shape)
                i+=1

        if test_sub in filelist[line-1] and ti>0 and not test_sub in filelist[line]:
            #test_y.append((int(filelist[len(filelist) - 1][-5]) - 1) / 4)
            test_X.pop()
        elif line ==len(filelist) and len(dataset_X)!= len(dataset_y):
            dataset_y.append((int(filelist[len(filelist) - 1][-5]) - 1) / 4)
        elif line == len(filelist) and len(test_X) != len(test_y):
            test_y.append((int(filelist[len(filelist) - 1][-5]) - 1) / 4)
    if(len(dataset_X>len(dataset_y))):
        dataset_X.pop()
    dataset_X= np.array(dataset_X)
    number, nchannel, nx, ny = dataset_X.shape
    dataset_X = dataset_X.reshape((number, nx , ny, nchannel))
    dataset_y=np.array(dataset_y)
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]
    test_X= np.array(test_X)
    number, nchannel, nx, ny = test_X.shape
    test_X = test_X.reshape((number, nx , ny, nchannel))
    return dataset_X, dataset_y,test_X,test_y

# Resnet(비슷한 거)
def image_CNN(X, y, x_test, y_test, test_sub):
        import tensorflow as tf
        from keras import Input
        from keras.models import Model
        from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
        tf.logging.set_verbosity(tf.logging.ERROR)
        # 1th layer
        input_tensor = Input(shape=shape_cnn, dtype='float32', name='input')
        K = 1

        def conv1_layer(x):
            x = ZeroPadding2D(padding=(3, 3))(x)
            x = Conv2D(64, (7, 7), strides=(2, 2))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = ZeroPadding2D(padding=(1, 1))(x)

            return x

        def conv2_layer(x):
            x = MaxPooling2D((3, 3), 2)(x)
            shortcut = x
            for i in range(3):
                if (i == 0):
                    x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                    shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
                    x = BatchNormalization()(x)
                    shortcut = BatchNormalization()(shortcut)

                    x = Add()([x, shortcut])
                    x = Activation('relu')(x)

                    shortcut = x

                else:
                    x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)

                    x = Add()([x, shortcut])
                    x = Activation('relu')(x)

                    shortcut = x

            return x

        def conv3_layer(x):
            shortcut = x

            for i in range(4):
                if (i == 0):
                    x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                    shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                    x = BatchNormalization()(x)
                    shortcut = BatchNormalization()(shortcut)

                    x = Add()([x, shortcut])
                    x = Activation('relu')(x)

                    shortcut = x

                else:
                    x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)

                    x = Add()([x, shortcut])
                    x = Activation('relu')(x)

                    shortcut = x

            return x

        x = conv1_layer(input_tensor)
        x = conv2_layer(x)
        x = conv3_layer(x)
        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(K, activation='sigmoid')(x)

        resnet18 = Model(input_tensor, output_tensor)
        #resnet18.summary()

        from sklearn import metrics
        from keras.models import load_model
        filename = 'model_ver6.h5'
        if model_is_rm ==1:
            resnet18.compile(loss='mse', optimizer='adam', metrics=['mae'])
            resnet18.fit(X, y, batch_size=128, epochs=1, validation_split=0.3)
            resnet18.save(filename)
            load_model = load_model(filename)
            pred_X= np.array(load_model.predict(x_test)).T[0]
            predscore=np.vstack((pred_X,y_test))
            np.savetxt('./predicts/subjects/predict_test_'+test_sub+'.csv',(predscore*4+1).T, fmt='%s', delimiter=",")
            print('savetxt==>/predicts/subjects/predict_test_'+test_sub+'.csv')
        else:
            resnet18.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            resnet18.fit(X, y, batch_size=128, epochs=1, validation_split=0.3)
            resnet18.save(filename)
            load_model = load_model(filename)
            pred_X = np.argmax(load_model.predict(x_test,verbose=0)*4+1)
            np.savetxt('./predicts/confusionmat_test_'+test_sub+'.csv', metrics.confusion_matrix(pred_X, y_test*4+1),
                       delimiter=',')


if __name__ == "__main__":
        test_sub_all = np.loadtxt('D:\\PycharmProjects\\EEG_signal_PSD_DE\\all_people.txt',dtype='str', delimiter=",")
    #for sub in range(171,184):
        sub= len(test_sub_all)-1
        data_train_x, data_train_y, data_x, data_y= dataset(dataset_x_dir, test_sub_all[sub])
        if (len(data_x) != len(data_y)):
            print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_y)))
        else:
            image_CNN(data_train_x, data_train_y,data_x,data_y, test_sub_all[sub])


