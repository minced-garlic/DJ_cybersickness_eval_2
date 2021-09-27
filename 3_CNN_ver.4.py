'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

import numpy as np
import time, os
from attention_model.attention_module import attach_attention_module
from attention_model import dataset_import
import keras
from keras.regularizers import l2
import tensorflow as tf
from keras import Input
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, add
from scipy.stats import spearmanr
from keras import backend as K


tf.logging.set_verbosity(tf.logging.ERROR)

dataset_dir = 'D:/PycharmProjects/EEG_signal_PSD_DE/'
dataset_dir = 'D:/2019_data_for_Cnn/'
dataset_x_dir= 'no_spectrogram_infoICA_56X56_npy/'

# Training parameters

y_dir ='C:\\Users\\Admin\\PycharmProjects\\EEG_signal_PSD_DE/Score_mean.csv'
#validation_split= 0.1
module_name= [None, 'cbam_block', 'se_block', 'channel_block']

epochs = 100
num_classes = 1
scores=None
scores= [0,0,1,2,3]
attention_module = module_name[1]
a=['standardization', 'normalization']
stand=a[0]

def _op(preds, actuals):
    rho, _ = spearmanr(preds, actuals)
    return rho

def correlation_coefficient(y_true, y_pred):
    return tf.py_func(_op, [y_pred,y_true], tf.float64)

def single_class_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        recall_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_recall_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * recall_mask
        class_recall = K.cast(K.sum(class_recall_tensor), 'float32') / K.cast(K.maximum(K.sum(recall_mask), 1) , 'float32')
        return class_recall
    return recall


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

def image_CNN(x_train, y_train, x_test, y_test, dat_inf):
        # 1th layer
        input_shape = x_train.shape[1:]

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        if stand =='normalization':
            x_train_max = np.max(x_train, axis=0)
            x_train_min = np.min(x_train, axis=0)
            x_train -= x_train_min
            x_train /= x_train_max-x_train_min
            x_test -= x_train_min
            x_test /= x_train_max-x_train_min
        elif stand =='standardization':
            x_train_std = np.std(x_train, axis=0)
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_train /= x_train_std*3
            x_test -= x_train_mean
            x_test /= x_train_std*3



        input_tensor = Input(shape=input_shape, dtype='float32', name='input')


        def conv1_layer(x):

            x = ZeroPadding2D(padding=(3, 3))(x)
            x = Conv2D(64, (7, 7), strides=(2, 2))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x_tmp = x
            x = ZeroPadding2D(padding=(1, 1))(x)

            return x, x_tmp

        def conv2_layer(x):
            num_filters, strides= 128, 1
            x = MaxPooling2D((3, 3), 2)(x)
            for i in range(3):
                if (i == 0):
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                    x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)


                    if attention_module is not None:
                        y = attach_attention_module(y, attention_module)
                    x = add([x, y])
                    x = Activation('relu')(x)


                else:
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                    x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                    if attention_module is not None:
                        y = attach_attention_module(y, attention_module)
                    x = add([x, y])
                    x = Activation('relu')(x)
            num_filters *= 2
            return x, num_filters, strides

        def conv3_layer(x, num_filters, strides):

            for i in range(3):
                if (i == 0):
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=2)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                    x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                    if attention_module is not None:
                        y = attach_attention_module(y, attention_module)
                    x = add([x, y])
                    x = Activation('relu')(x)
                else:
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                    x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)

                    x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                    if attention_module is not None:
                        y = attach_attention_module(y, attention_module)
                    x = add([x, y])
                    x = Activation('relu')(x)
            return x


        x, x_vis = conv1_layer(input_tensor)
        x, num_filters, strides = conv2_layer(x)
        x = conv3_layer(x,num_filters, strides)
        x = GlobalAveragePooling2D()(x)
        output_tensor = Dense(num_classes, activation='sigmoid')(x)

        resnet18 = Model(input_tensor, output_tensor)
        time_start= time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))
        from keras.callbacks import ModelCheckpoint
        print('attention_module:', attention_module)
        print('dataset:', dataset_x_dir[:-1])
        print('epochs:', epochs)
        print('num_classes:', num_classes)
        print('stand:', stand)
        print('SCORES:', scores)

        if num_classes == 1:
            y_train = (y_train.astype('float32')) / np.max(y_train, axis=0)
            y_test = (y_test.astype('float32')) / np.max(y_train, axis=0)
            y_train= tf.convert_to_tensor(y_train, np.float32)
            y_test= tf.convert_to_tensor(y_test, np.float32)
            resnet18.compile(loss='mse', optimizer='adam', metrics=['mse',  correlation_coefficient])
            #resnet18.summary()
            filename = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models/model_ver' + time_start + '(epochs_' + str(epochs) + '_num_classes_' + str(
                num_classes) + ').json'
            mc = ModelCheckpoint(filename, monitor='val_correlation_coefficient_loss', mode='max', save_best_only=True)
            resnet18.fit(x_train, y_train, steps_per_epoch=10, validation_steps=2, epochs=epochs, validation_data=(x_test, y_test),
                          callbacks=[mc])

            if not os.path.isfile(filename):
                print(filename,'does not exist')
                model_json= resnet18.to_json()
                with open(filename, "w") as json_file:
                    json_file.write(model_json)

            from keras.models import model_from_json
            json_file = open(filename, "r")
            loaded_model_json = json_file.read()
            json_file.close()
            load_resnet = model_from_json(loaded_model_json, encoding='utf-8')

            pred_X = np.array(load_resnet.predict(x_test)).T[0]
            pred_train_X = np.array(load_resnet.predict(x_train)).T[0]
            predscore = np.vstack((pred_X, y_test)).T * 4 + 1
            predtrain = np.vstack((pred_train_X, y_train)).T* 4 + 1
            if attention_module is not None and dat_inf is not None:
                predscore = np.hstack((dat_inf[1],predscore))
                predtrain = np.hstack((dat_inf[0],predtrain))
                np.savetxt(
                    './predicts/predict_test_' + attention_module + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                        epochs) + '_num_classes_' + str(num_classes) + ').csv', predscore, fmt='%s',
                    delimiter=",")
                np.savetxt(
                    './predicts/predict_score_' + attention_module + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                        epochs) + '_num_classes_' + str(num_classes) + ').csv', predtrain, fmt='%s',
                    delimiter=",")
            elif attention_module is not None:
                np.savetxt('./predicts/predict_test_'+ attention_module + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                    epochs) + '_num_classes_' + str(num_classes) + ').csv', (predscore * 4 + 1).T, fmt='%.3f',
                           delimiter=",")
                np.savetxt('./predicts/predict_score_' +attention_module+ dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                    epochs) + '_num_classes_' + str(num_classes) + ').csv', (predtrain * 4 + 1).T, fmt='%.3f',
                           delimiter=",")
            elif dat_inf is not None:
                predscore = np.hstack((dat_inf[1],predscore))
                predtrain = np.hstack((dat_inf[0],predtrain))
                np.savetxt(
                    './predicts/predict_test_' + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                        epochs) + '_num_classes_' + str(num_classes) + ').csv', predscore, fmt='%s',
                    delimiter=",")
                np.savetxt(
                    './predicts/predict_score_' + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                        epochs) + '_num_classes_' + str(num_classes) + ').csv', predtrain, fmt='%s',
                    delimiter=",")

            else:
                np.savetxt('./predicts/predict_test_' + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                    epochs) + '_num_classes_' + str(num_classes) + ').csv', (predscore * 4 + 1).T, fmt='%.3f',
                           delimiter=",")
                np.savetxt('./predicts/predict_score_' + dataset_x_dir[:-3] + time_start + '(epochs_' + str(
                    epochs) + '_num_classes_' + str(num_classes) + ').csv', (predtrain * 4 + 1).T, fmt='%.3f',
                           delimiter=",")
        elif num_classes > 1:

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            resnet18.compile(loss='categorical_crossentropy', optimizer='sgd',
                             metrics= [keras.metrics.categorical_accuracy,
                                       single_class_recall(0),single_class_recall(1),single_class_recall(2),
                                       single_class_recall(3),single_class_recall(4)])
            from sklearn import metrics
            #resnet18.summary()

            filename4 = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models/model_ver' + \
                        time_start + '_best4_(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').json'
            filename5 = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2//models/model_ver' + \
                        time_start + '_best5_(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').json'
            mc4= ModelCheckpoint(filepath=filename4, monitor='val_recall_4', mode='max', save_best_only=True)
            mc5 =ModelCheckpoint(filepath=filename5, monitor='val_recall_3', mode='max', save_best_only=True)

            resnet18.fit(x_train, y_train, steps_per_epoch=10, validation_steps=2,  epochs=epochs,
                    validation_data = (x_test, y_test), class_weight = { 0.1, 0.2,0.1,0.4,0.5},callbacks=[mc4,mc5])

            load_model = resnet18
            pred_X = load_model.predict(x_test)
            pred_X= np.array(pred_X.argmax(axis=-1)).T
            pred_train_X=load_model.predict(x_train)
            pred_train_X = np.array(pred_train_X.argmax(axis=-1)).T
            y_test = y_test.argmax(axis=-1).T
            y_train = y_train.argmax(axis=-1).T
            if attention_module is not None:
                print(metrics.confusion_matrix(pred_X, y_test))
                np.savetxt('./predicts/confusionmat_test_' +attention_module+ time_start + '(epochs_' + str(epochs) + ').csv',
                           metrics.confusion_matrix(pred_X, y_test),
                           delimiter=',', fmt='%i')
                np.savetxt('./predicts/confusionmat_train_' +attention_module+ time_start + '(epochs_' + str(epochs) + ').csv',
                           metrics.confusion_matrix(pred_train_X, y_train),
                           delimiter=',', fmt='%i')
            else:
                np.savetxt('./predicts/confusionmat_test_' + time_start + '(epochs_' + str(epochs) + ').csv',
                           metrics.confusion_matrix(pred_X, y_test),
                           delimiter=',',fmt='%i')
                np.savetxt('./predicts/confusionmat_train_' + time_start + '(epochs_' + str(epochs) + ').csv',
                           metrics.confusion_matrix(pred_train_X, y_train),
                           delimiter=',',fmt='%i')

        else:
            raise Exception("'{}' is not supported attention module!".format(num_classes))


if __name__ == "__main__":
    data_train_x, data_train_y, data_x, data_y, dat_inf= dataset_import.dataset(dataset_dir+dataset_x_dir, train=4, test=1, testing= False, info= False, scores= scores)
    if (len(data_x) != len(data_y)):
        print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_y)))
    else:
        image_CNN(data_train_x, data_train_y,data_x,data_y, dat_inf)


