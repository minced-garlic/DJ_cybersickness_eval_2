'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

import numpy as np
import os, cv2, time
import matplotlib.pyplot as plt


dataset_x_dir = 'D:/PycharmProjects/EEG_signal_PSD_DE/spectrogram_infoICA(CNN2)\\'

y_dir ='C:\\Users\\Admin\\PycharmProjects\\EEG_signal_PSD_DE/Score_mean.csv'
train = 4
test = 1
shape_cnn =(56,56,8)
epochs = 1
model_is_rm = 1



def dataset(dataset_x_dir):
    dataset_X = [[]]
    dataset_y = []
    dataset_label=[]
    filelist = os.listdir(dataset_x_dir)
    filelist.sort()
    i=0
    for line in range(len(filelist)):
        img = cv2.imread(dataset_x_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        dataset_X[i].append(img)
        if line<len(filelist)-1 and filelist[line][:-8] != filelist[line + 1][:-8]:
            dataset_X.append([])
            '''
            if int(filelist[line][-5])>3:
                dataset_y.append(1)
            else:
                dataset_y.append((int(filelist[line][-5]) - 1) / 4)
            '''
            dataset_y.append((int(filelist[line][-5])-1)/4)
            dataset_label.append([filelist[line][:7],filelist[line][23:27] ])
            if np.array(dataset_X[i]).shape != np.array(dataset_X[i-1]).shape and i>1:
                print(i,'번 데이터가 다름:' ,np.array(dataset_X[i]).shape)
            i+=1

    dataset_y.append((int(filelist[len(filelist)-1][-5])-1)/4)
    dataset_label.append([filelist[len(filelist)-1][:7],filelist[len(filelist)-1][23:27]])
    dataset_X= np.array(dataset_X)
    number, nchannel, nx, ny = dataset_X.shape
    dataset_X = dataset_X.reshape((number, nx , ny, nchannel))
    dataset_y=np.array(dataset_y)
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]
    dataset_label=np.array(dataset_label)
    dataset_label=dataset_label[s]

    return dataset_X, dataset_y, dataset_label

# 이 뭐같은 dataset 3은 평균치를 끌어와 데이터를 매겨봅니다... 하핫!

def dataset3(dataset_x_dir, train, test):
    dataset_X = [[]]
    dataset_y = []
    filelist = os.listdir(dataset_x_dir)
    filelist.sort()
    i=0
    for line in range(len(filelist)):
        img = cv2.imread(dataset_x_dir+filelist[line], cv2.IMREAD_GRAYSCALE)
        dataset_X[i].append(img)
        if line<len(filelist)-1 and filelist[line][:-8] != filelist[line + 1][:-8]:
            dataset_X.append([])

            Y_all = np.loadtxt(y_dir, dtype=str, delimiter=',')
            dataset_y.append((float([Yline[4] for Yline in Y_all if filelist[line][-14:-10] == Yline[0] ][0])-2)/2)

            if np.array(dataset_X[i]).shape != np.array(dataset_X[i-1]).shape and i>1:
                print(i,'번 데이터가 다름:' ,np.array(dataset_X[i]).shape)
            i+=1

    dataset_y.append(int(filelist[len(filelist)-1][-5]))
    dataset_X= np.array(dataset_X)
    number, nchannel, nx, ny = dataset_X.shape
    dataset_X = dataset_X.reshape((number, nx , ny, nchannel))
    dataset_y=np.array(dataset_y)
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]


    return dataset_X[int(number*test/(train+test)):], dataset_y[int(number*test/(train+test)):], dataset_X[:int(number*test/(train+test))], dataset_y[:int(number*test/(train+test))]

# Resnet(비슷한 거)
def image_CNN(X, y,dataset_label):
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

            for i in range(3):
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
        resnet18.summary()

        time_start= time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))

        from keras.models import load_model
        if model_is_rm ==1:
            resnet18.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
            resnet18.fit(X, y, batch_size=128, epochs=epochs, validation_split=0.1)
            filename = 'model_ver2.h5'
            resnet18.save(filename)
            load_model = load_model(filename)
            pred_train_X = np.array(load_model.predict(X)).T[0]
            predtrain= np.vstack((pred_train_X,y))*4+1
            predtrain = np.vstack((dataset_label,predtrain.T))
            np.savetxt('./predicts/predict_score'+time_start+'(epochs_'+str(epochs)+').csv',predtrain, fmt='%s', delimiter=",")
        else:
            resnet18.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            resnet18.fit(X, y, batch_size=128, epochs=1, validation_split=0.1)
            filename = 'model_ver2.h5'
            from sklearn import metrics
            resnet18.save(filename)
            load_model = load_model(filename)
            pred_train_X = np.argmax(load_model.predict(X,verbose=0))
            predtrain= np.vstack((pred_train_X,y))*4+1
            predtrain = np.vstack((dataset_label,predtrain))
            np.savetxt('./predicts/predict_score'+time_start+'(epochs_'+str(epochs)+').csv',predtrain.T, fmt='%s', delimiter=",")
            np.savetxt('./predicts/confusionmat_train'+time_start+'(epochs_'+str(epochs)+').csv', metrics.confusion_matrix(pred_train_X, y),
                       delimiter=',')


if __name__ == "__main__":
    data_x, data_y, dataset_label= dataset(dataset_x_dir)
    if (len(data_x) != len(data_y)):
        print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_y)))
    else:
        image_CNN(data_x, data_y,dataset_label)


