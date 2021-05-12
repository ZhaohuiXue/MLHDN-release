# model script
"""
@author: mengxue.zhang
"""


from keras.models import Model
from keras.optimizers import Adam
from keras import losses
from keras.layers import Input, Concatenate, Lambda, Flatten, Dense, Add, Conv2D
import keras.backend as K

from mvsecondpooling import MVSecondOrderPooling
from clrbp import CLRBP, LRBP


def cal_input_shape(d=[1, 3, 9, 27]):
    dilation_cut = range(0, 100, 2)
    shp = 3
    for i in d:
        shp += dilation_cut[i]
    return [shp, shp]


def get_model(classes=10, lr=1e-3, name='MLHDN', image_shape=[83, 83],
              hyperparams={'r1': 4, 'r2': 1, 't': 'mlrbp', 'v': 4, 'd': [1, 3, 9, 27], 's': 1}):
    while True:
        view = hyperparams['v']
        if name == 'HDN':
            r1 = hyperparams['r1']
            r2 = hyperparams['r2']
            type = hyperparams['t']
            dilation_rates = hyperparams['d']
            model = HDN(width=image_shape[0], height=image_shape[1], channel=view, classes=classes,
                        rates=dilation_rates, type=type, r1=r1, r2=r2)
        else:
            r1 = hyperparams['r1']
            r2 = hyperparams['r2']
            type = hyperparams['t']
            dilation_rates = hyperparams['d']
            model = MLHDN(width=image_shape[0], height=image_shape[1], channel=view, classes=classes,
                           rates=dilation_rates, type=type, r1=r1, r2=r2)

        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        yield model


def HDN(width, height, channel, classes, rates=[1, 3, 9, 27], type='mlrbp', r1=2, r2=1):
    return MLHDN(width, height, channel, classes, rates=rates, type=type, r1=r1, r2=r2, c=1)


def MLHDN(width, height, channel, classes, rates=[1, 3, 9, 27], type='mlrbp', r1=2, r2=1, c=-1):
    if c == -1:
        c = channel

    inpt = Input(shape=(width, height, channel))
    conv1 = Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', name='c1', dilation_rate=rates[0])
    conv2 = Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', name='c2', dilation_rate=rates[1])
    conv3 = Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu', name='c3', dilation_rate=rates[2])
    conv4 = Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu', name='c4', dilation_rate=rates[3])

    lrc = LRBP(r=8, class_num=classes, view=c)
    mlrc = CLRBP(r=r1, r2=r2, view=c, class_num=classes)
    fc = Dense(units=classes, activation='softmax')

    add = Add()
    concat = Concatenate(axis=-1)
    sop = MVSecondOrderPooling(view=c, name='bp_feature')

    re = Lambda(lambda x: K.reshape(x, shape=[-1, K.int_shape(x)[1] * K.int_shape(x)[2], K.int_shape(x)[3]]))
    norm1 = Lambda(lambda x: K.l2_normalize(x, axis=-1))
    norm2 = Lambda(lambda x: K.l2_normalize(x, axis=[-2, -1]), name='second_order_feature')
    flatten = Flatten()

    # Multi-View Feature Extraction
    f = []
    for i in range(c):
        xi = Lambda(lambda x: K.expand_dims(x[:, :, :, i], axis=-1))(inpt)
        fi = conv4(conv3(conv2(conv1((xi)))))
        fi = re(fi)
        f.append(fi)

    # Multi-View Feature Fusion
    if type == 'add':
        oupt = fc(norm1(flatten(add(f))))
    elif type == 'concat':
        oupt = fc(norm1(flatten(concat(f))))
    elif type == 'bp':
        oupt = fc(norm1(flatten(sop(f))))
    elif type == 'lrbp':
        oupt = lrc(norm2(sop(f)))
    else:
        oupt = mlrc(norm2(sop(f)))

    models = Model(inputs=inpt, outputs=oupt)
    return models
