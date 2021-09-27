'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

import numpy as np
import time, os
from attention_model import dataset_import,cnn_Keras_ver_1,SVM
import keras
import tensorflow as tf
from keras.models import Model
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
num_classes = 5
scores=None
#scores= [0,0,1,2,3]
attention_module = module_name[0]
a=['standardization', 'normalization']
stand=a[0]


def single_class_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        recall_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_recall_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * recall_mask
        class_recall = K.cast(K.sum(class_recall_tensor), 'float32') / K.cast(K.maximum(K.sum(recall_mask), 1) , 'float32')
        return class_recall
    return recall


def _op(preds, actuals):
    rho, _ = spearmanr(preds, actuals)
    return rho

def correlation_coefficient(y_true, y_pred):
    return tf.py_func(_op, [y_pred,y_true], tf.float64)


def image_CNN(x_train, x_test, ps_train,  ps_test, sc_train,  sc_test, score_train,score_test,categorizes=None):
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

        time_start= time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))
        from keras.callbacks import ModelCheckpoint
        print('attention_module:', attention_module)
        print('dataset:', dataset_x_dir[:-1])
        print('epochs:', epochs)
        print('num_classes:', num_classes)
        print('stand:', stand)
        print('SCORES:', scores)

        if num_classes == 1:
            ps_train = (ps_train.astype('float32')) / np.max(ps_train, axis=0)
            ps_test = (ps_test.astype('float32')) / np.max(ps_train, axis=0)
            ps_train= tf.convert_to_tensor(ps_train, np.float32)
            ps_test= tf.convert_to_tensor(ps_test, np.float32)
            sc_train = (sc_train.astype('float32')) / np.max(sc_train, axis=0)
            sc_test = (sc_test.astype('float32')) / np.max(sc_train, axis=0)
            sc_train= tf.convert_to_tensor(sc_train, np.float32)
            sc_test= tf.convert_to_tensor(sc_test, np.float32)
            from keras.callbacks import EarlyStopping

            resnet18 = cnn_Keras_ver_1.cnn_keras(x_train.shape[1:], num_classes=num_classes,
                                                 attention_module=attention_module)
            resnet18.compile(loss='mse', optimizer='adam', metrics=['mse',  correlation_coefficient])
            resnet18.summary()
            filename_1 = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models/modelps_ver' + \
                         time_start + '(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').h5'
            filename_2 = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models/modelsc_ver' + \
                         time_start + '(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').h5'

            mc1 = ModelCheckpoint(filename_1, monitor='val_correlation_coefficient', mode='max', save_best_only=True)
            mc2 = ModelCheckpoint(filename_2, monitor='val_correlation_coefficient', mode='max', save_best_only=True)
            es = EarlyStopping(monitor='categorical_accuracy', mode='max', baseline=1.0)

            resnet18.fit(x_train, ps_train, steps_per_epoch=10, validation_steps=2, epochs=epochs, validation_data=(x_test, ps_test),
                          callbacks=[mc1,es ])
            model1 = Model(inputs=resnet18.inputs, outputs=resnet18.layers[len(resnet18.layers) - 2].output)
            resnet18.fit(x_train, sc_train, steps_per_epoch=10, validation_steps=2, epochs=epochs, validation_data=(x_test, sc_test),
                          callbacks=[mc2,es ])
            model2 = Model(inputs=resnet18.inputs, outputs=resnet18.layers[len(resnet18.layers) - 2].output)

            from sklearn import metrics
            from sklearn.svm import SVC
            clf = SVC(kernel='rbf', gamma='scale', C=15.0)
            feature_map1 = model1.predict(x_train)
            feature_map2 = model2.predict(x_train)
            feature_map_train= np.hstack((feature_map1,feature_map2))
            SVMmodel = clf.fit(feature_map_train, score_train)
            feature_map1 = model1.predict(x_test)
            feature_map2 = model2.predict(x_test)
            feature_map_test= np.hstack((feature_map1,feature_map2))

            print('feature_map1.shape',feature_map1.shape)
            print('feature_map2.shape',feature_map2.shape)

        elif num_classes > 1:

            ps_train = keras.utils.to_categorical(ps_train, categorizes[0]+1)
            ps_test = keras.utils.to_categorical(ps_test, categorizes[0]+1)
            sc_train = keras.utils.to_categorical(sc_train, categorizes[1]+1)
            sc_test = keras.utils.to_categorical(ps_test, categorizes[1]+1)

            resnet18 = cnn_Keras_ver_1.cnn_keras(x_train.shape[1:], num_classes=categorizes[0]+1,
                                                 attention_module=attention_module)
            resnet18.compile(loss='categorical_crossentropy', optimizer='sgd',
                             metrics= [keras.metrics.categorical_accuracy,
                                       single_class_recall(0),single_class_recall(1),single_class_recall(2),
                                       single_class_recall(3),single_class_recall(4)])
            from sklearn import metrics
            resnet18.summary()

            filename1 = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models/model_ver' + \
                        time_start + '_1(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').json'
            filename2 = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2//models/model_ver' + \
                        time_start + '_2(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').json'
            mc2= ModelCheckpoint(filepath=filename2, monitor='val_categorical_accuracy', mode='max', save_best_only=True)
            mc1 =ModelCheckpoint(filepath=filename1, monitor='val_categorical_accuracy', mode='max', save_best_only=True)

            model1 = Model(inputs=resnet18.inputs, outputs=resnet18.layers[len(resnet18.layers) - 2].output)
            resnet18.fit(x_train, ps_train, steps_per_epoch=10, validation_steps=2, epochs=epochs, validation_data=(x_test, ps_test),
                          callbacks=[mc1])
            resnet18 = cnn_Keras_ver_1.cnn_keras(x_train.shape[1:], num_classes=categorizes[1]+1,
                                                 attention_module=attention_module)
            resnet18.compile(loss='categorical_crossentropy', optimizer='sgd',
                             metrics= [keras.metrics.categorical_accuracy,
                                       single_class_recall(0),single_class_recall(1),single_class_recall(2),
                                       single_class_recall(3),single_class_recall(4)])
            resnet18.fit(x_train, sc_train, steps_per_epoch=10, validation_steps=2, epochs=epochs, validation_data=(x_test, sc_test),
                          callbacks=[mc2])
            model2 = Model(inputs=resnet18.inputs, outputs=resnet18.layers[len(resnet18.layers) - 2].output)

            feature_map1 = model1.predict(x_test)
            feature_map2 = model2.predict(x_test)
            print('feature_map1.shape',feature_map1.shape)
            print('feature_map2.shape',feature_map2.shape)

            from sklearn import metrics
            from sklearn.svm import SVC
            clf = SVC(kernel='rbf', gamma='scale', C=15.0)
            feature_map1 = model1.predict(x_train)
            feature_map2 = model2.predict(x_train)
            feature_map_train= np.hstack((feature_map1,feature_map2))
            SVMmodel = clf.fit(feature_map_train, score_train)
            feature_map1 = model1.predict(x_test)
            feature_map2 = model2.predict(x_test)
            feature_map_test= np.hstack((feature_map1,feature_map2))

            print('feature_map1.shape',feature_map1.shape)
            print('feature_map2.shape',feature_map2.shape)
            print("훈련 세트 정확도: {:.4f}".format(SVMmodel.score(feature_map_train, score_train)))
            print("테스트 세트 정확도: {:.4f}".format(SVMmodel.score(feature_map_test, score_test)))
            print("훈련 세트 정확도:\n", metrics.confusion_matrix(SVMmodel.predict(feature_map_train), score_train))
            print("테스트 세트 정확도:\n", metrics.confusion_matrix(SVMmodel.predict(feature_map_test),  score_test))
            np.savetxt('confusionmat_test.csv', metrics.confusion_matrix(SVMmodel.predict(feature_map_test), score_test),
                       delimiter=',')
            np.savetxt('confusionmat_train.csv', metrics.confusion_matrix(SVMmodel.predict(feature_map_train),score_train),
                       delimiter=',')


        else:
            raise Exception("'{}' is not supported attention module!".format(num_classes))

if __name__ == "__main__":
    if num_classes>1:
        data_t_x, data_x, data_t_ps, data_ps, data_t_scene, data_scene,score_train ,score_test, categorizes,= \
            dataset_import.dataset_label_n_score2(dataset_dir+dataset_x_dir,num_classes=num_classes)
    else:
        data_t_x, data_x, data_t_ps, data_ps, data_t_scene, data_scene,data_score= \
            dataset_import.dataset_label_n_score(dataset_dir+dataset_x_dir,num_classes=num_classes)
    if (len(data_x) != len(data_ps)):
        print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(data_ps)))
    else:
        image_CNN(data_t_x, data_x, data_t_ps, data_ps, data_t_scene, data_scene,score_train ,score_test ,categorizes)


