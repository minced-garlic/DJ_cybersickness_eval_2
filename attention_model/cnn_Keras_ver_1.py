
from .attention_module import attach_attention_module
from keras.regularizers import l2
from keras import Input
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, add, Dropout
from scipy.stats import spearmanr
from keras.models import Model
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

def correlation_coefficient_loss(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                       tf.cast(y_true, tf.float32)], Tout=tf.float32))

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



def conv1_layer(x,attention_module=None):

    num_filters, strides = 128, 1
    if attention_module is 'scan_block':
        y = resnet_layer(inputs=x,
                         num_filters=num_filters,
                         strides=strides)
        y = resnet_layer(inputs=y,
                         num_filters=num_filters,
                         activation=None)
        y = attach_attention_module(x, attention_module)
        x = add([x, y])
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    '''
    if attention_module is 'scan_block':
        y = attach_attention_module(x, attention_module)
        x = add([x, y])
    
    '''
    x_tmp = x
    x = ZeroPadding2D(padding=(1, 1))(x)
    return x, x_tmp


def conv2_layer(x,attention_module=None):
    num_filters, strides = 128, 1
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

            if attention_module is not None and attention_module is not 'scan_block':
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
            if attention_module is not None and attention_module is not 'scan_block':
                y = attach_attention_module(y, attention_module)
            x = add([x, y])
            x = Activation('relu')(x)
    num_filters *= 2
    return x, num_filters, strides


def conv3_layer(x, num_filters, strides,attention_module=None):
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
            if attention_module is not None and attention_module is not 'scan_block':
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
            if attention_module is not None and attention_module is not 'scan_block':
                y = attach_attention_module(y, attention_module)
            x = add([x, y])
            x = Activation('relu')(x)
    return x

def cnn_keras(input_shape, num_classes, attention_module=None):
    input_tensor = Input(shape=input_shape, dtype='float32', name='input')

    x, x_vis = conv1_layer(input_tensor)
    x, num_filters, strides = conv2_layer(x,attention_module=attention_module)
    x = conv3_layer(x, num_filters, strides,attention_module=attention_module)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_classes, activation='sigmoid')(x)

    model = Model(input_tensor, output_tensor)
    return model



def cnn_keras_SVM(input_shape, num_classes, attention_module=None):

    #for input_shape in input_X:
    input_tensor = Input(shape=input_shape, dtype='float32', name='input')
    x, x_vis = conv1_layer(input_tensor,attention_module=attention_module)
    x, num_filters, strides = conv2_layer(x,attention_module=attention_module)
    x = conv3_layer(x, num_filters, strides,attention_module=attention_module)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(256, activation='sigmoid')(x)

    model = Model(input_tensor, output_tensor)
    return model


def jeong_layer(input_shape, num_classes, attention_module=None):

    input_tensor = Input(shape=input_shape, dtype='float32', name='input')
    x = Dense(84, kernel_initializer='uniform', activation='relu')(input_tensor)
    x= Dropout(0.25)(x)

    x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
    x= Dropout(0.25)(x)

    x = Dense(256, kernel_initializer='uniform', activation='relu')(x)
    x= Dropout(0.25)(x)

    x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
    x= Dropout(0.25)(x)
    x = Dense(32, activation='sigmoid')(x)
    x= Dropout(0.25)(x)
    output_tensor = Dense(num_classes, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)
    return model


