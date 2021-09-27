'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

import numpy as np
import os, cv2, time
import matplotlib.pyplot as plt


dataset_x_dir= 'D:\\PycharmProjects\\data_cut_by_0.1_infoICA\\'

dataset_y_dir ='D:\\PycharmProjects\\opticalflow_4X4\\'
train = 4
test = 1
shape_cnn =(None, 32,32)
epochs = 10
model_is_rm = 1


def dataset(dataset_x_dir, train, test):
    dataset_X, score_X, dataset_y = [],[],[]
    filelistY = os.listdir(dataset_y_dir)
    filelistY.sort()
    #for line in range(100000):
    for line in range(len(filelistY)):
        if os.path.isfile(dataset_x_dir+filelistY[line]) and os.path.isfile(dataset_y_dir+filelistY[line]):
            x= np.loadtxt(dataset_x_dir+filelistY[line], delimiter=',')
            if x.shape == (8,128):
                x = x.reshape((32, 32))
                dataset_X.append(x)
                dataset_y.append(np.loadtxt(dataset_y_dir+filelistY[line], delimiter=','))

    dataset_X=np.array(dataset_X)
    dataset_X = dataset_X.reshape((-1,1,32,32))
    dataset_y=np.array(dataset_y)
    s = np.arange(np.array(dataset_y).shape[0])
    dataset_y = dataset_y.reshape((len(s),-1))
    np.random.shuffle(s)
    dataset_X= dataset_X[s]
    dataset_y= dataset_y[s]
    return dataset_X[int(test/(train+test)):], dataset_y[int(test/(train+test)):], dataset_X[:int(test/(train+test))], dataset_y[:int(test/(train+test))]


# Resnet(비슷한 거)
def image_CNN(X, y, x_test, y_test):
        import tensorflow as tf
        from keras import Input
        from keras.models import Model
        from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
        tf.logging.set_verbosity(tf.logging.ERROR)
        # 1th layer
        input_tensor = Input(shape=shape_cnn, dtype='float32', name='input')


        def conv1_layer(x):
            x = ZeroPadding2D(padding=(3, 3))(x)
            x = Conv2D(32, (5, 5), strides=(2, 2))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = ZeroPadding2D(padding=(1, 1))(x)

            return x

        def conv2_layer(x):
            shortcut = x

            for i in range(3):
                if (i == 0):
                    x = Conv2D(64, (1, 1), strides=(2, 2), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
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
                    x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
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
        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(32, activation='sigmoid')(x)

        resnet18 = Model(input_tensor, output_tensor)
        resnet18.summary()

        time_start= time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))
        if model_is_rm ==1:

            resnet18.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
            hist = resnet18.fit(X, y, batch_size=128, epochs=epochs, validation_split=0.3)
            filename = './models/model_ver'+time_start+'(epochs_'+str(epochs)+').h5'
            from sklearn import metrics
            from keras.models import load_model
            resnet18.save(filename)
            load_model = load_model(filename)
            score = load_model.evaluate(x_test, y_test, verbose=0)
            print(score)
        else:
            resnet18.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            hist = resnet18.fit(X, y, batch_size=128, epochs=1, validation_split=0.1)
            filename = 'model_ver'+time_start+'(epochs_'+str(epochs)+').h5'
            from sklearn import metrics
            from keras.models import load_model
            resnet18.save(filename)
            load_model = load_model(filename)
            pred_X = np.argmax(load_model.predict(x_test,verbose=0))
            pred_train_X = np.argmax(load_model.predict(X,verbose=0))
            np.savetxt('./predicts/confusionmat_test'+time_start+'(epochs_'+str(epochs)+').csv', metrics.confusion_matrix(pred_X, y_test),
                       delimiter=',')
            np.savetxt('./predicts/confusionmat_train'+time_start+'(epochs_'+str(epochs)+').csv', metrics.confusion_matrix(pred_train_X, y),
                       delimiter=',')



if __name__ == "__main__":
    data_train_x, data_train_y, data_x, data_y= dataset(dataset_x_dir, train, test)
    if (len(data_x) != len(data_y)):
        print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_y)))
    else:
        image_CNN(data_train_x, data_train_y,data_x,data_y)


