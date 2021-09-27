'''
def dataset_x: >> dataset 만드는 함수: (224,224,3, 8)
: 224X224(이미지 크기)X8(1-8 eeg 채널 수)
def dataset_y:

'''

import numpy as np
import time
from attention_model import dataset_import,cnn_Keras_ver_1
import keras,platform
import tensorflow as tf
from keras.models import Model
from scipy.stats import spearmanr
from keras import backend as K


print(platform.architecture())

tf.logging.set_verbosity(tf.logging.ERROR)

dataset_dir = 'D:/PycharmProjects/EEG_signal_PSD_DE/'
dataset_dir = 'D:/2019_data_for_Cnn/'
dataset_x_dir= 'no_spec_infoICA_56X56_npy_percent/'
#dataset_x_dir='testing100/'
module_name= [None, 'cbam_block', 'se_block', 'channel_block','scan_block']
epochs = 100
num_classes = 5
scores=None
#scores= [0,0,1,2,3]
attention_module = module_name[4]
a=['standardization', 'normalization', None]
stand=a[2]
testing= False
average_score=False
from keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

def single_class_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        recall_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_recall_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * recall_mask
        class_recall = K.cast(K.sum(class_recall_tensor), 'float32') / K.cast(K.maximum(K.sum(recall_mask), 1) , 'float32')
        return class_recall
    return recall

def data_nor_or_std(x_train,x_test, stand):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if stand == 'normalization':
        x_train_max = np.max(x_train, axis=0)
        x_train_min = np.min(x_train, axis=0)
        x_train -= x_train_min
        x_train /= x_train_max - x_train_min
        x_test -= x_train_min
        x_test /= x_train_max - x_train_min
    elif stand == 'standardization':
        x_train_std = np.std(x_train, axis=0)
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_train /= x_train_std * 3
        x_test -= x_train_mean
        x_test /= x_train_std * 3
    return x_train,x_test

def regrassion_scaling(train, test):
    train = (train.astype('float32')) / np.max(train, axis=0)
    test = (test.astype('float32')) / np.max(train, axis=0)
    train = tf.convert_to_tensor(train, np.float32)
    test = tf.convert_to_tensor(test, np.float32)
    return train, test


def classification_scaling():
    return 0

def _op(preds, actuals):
    rho, _ = spearmanr(preds, actuals)
    return rho

def correlation_coefficient(y_true, y_pred):
    return tf.py_func(_op, [y_pred,y_true], tf.float64)


def image_CNN1(x_train, x_test, score_train,score_test,dataset_inf):

        x_train, x_test= data_nor_or_std(x_train,x_test,stand=stand) # 노말라이즈 or 스탠다리재이션

        time_start= time.strftime('_%Y_%m_%d_%H_%M', time.localtime(time.time()))
        print('attention_module:', attention_module)
        print('dataset:', dataset_x_dir[:-1])
        print('epochs:', epochs)
        print('num_classes:', num_classes)
        print('stand:', stand)
        print('SCORES:', scores)

        from keras.callbacks import ModelCheckpoint
        if num_classes > 1:
            score_train = keras.utils.to_categorical(score_train, num_classes)
            score_test = keras.utils.to_categorical(score_test, num_classes)

            resnet18 = cnn_Keras_ver_1.jeong_layer(x_train.shape[1:], num_classes=num_classes)
            resnet18.compile(loss='categorical_crossentropy', optimizer='sgd',
                             metrics= [keras.metrics.categorical_accuracy])
            from sklearn import metrics
            resnet18.summary()

            filename = 'C:\\Users\\Admin\\PycharmProjects\\DJ_cybersickness_eval_2/models/model_ver' + \
                        time_start + '_1(epochs_' + str(epochs) + '_num_classes_' + str(num_classes) + ').h5'
            mc =ModelCheckpoint(filepath=filename, monitor='val_categorical_accuracy', mode='max', save_best_only=True)
            es=  TerminateOnBaseline(monitor='categorical_accuracy', baseline=1.0)
            resnet18.fit(x_train, score_train, steps_per_epoch=10, validation_steps=2,  epochs=epochs,
                    validation_data = (x_test, score_test), callbacks=[mc,es])

            load_model = resnet18
            pred_X = load_model.predict(x_test)
            pred_X= np.array(pred_X.argmax(axis=-1)).T
            pred_train_X=load_model.predict(x_train)
            pred_train_X = np.array(pred_train_X.argmax(axis=-1)).T
            y_test = score_test.argmax(axis=-1).T
            y_train = score_train.argmax(axis=-1).T
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

        data_t_x, data_x, score_train ,score_test, dataset_inf= \
            dataset_import.dataset(dataset_dir+dataset_x_dir,scores=scores, testing = testing)
        data_t_x=data_t_x.reshape(len(score_train),-1)
        data_x = data_x.reshape(len(score_test), -1)
        if (len(data_x) != len(score_test)):
            print('data x(test x)의 길이 {}와 data y(test y)의 길이 {}가 같지 않습니다.'.format(len(data_x), len(score_test)))
        else:
            image_CNN1(data_t_x, data_x, score_train ,score_test, dataset_inf)



