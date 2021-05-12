# -*- coding: utf-8 -*-
"""
@author: mengxue.Zhang

@usage:
Y = CLRBP(r=r1, r2=r2, view=c, class_num=classes)(X) or
model.add(CLRBP(r=r1, r2=r2, view=c, class_num=classes))

X: a three-order tensor with a shape of [batch_size, W*H, C]
Y: a two-order tensor with a shape of [batch_size, class_num]

"""

from keras.layers import Layer
import keras.backend as K


class LRBP(Layer):
    def __init__(self,
                 r=6,
                 class_num=10,
                 view=4,
                 **kwargs):
        super(LRBP, self).__init__(**kwargs)
        self.r = r
        self.c_num = class_num
        self.view = view

    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.c_num, input_shape[-1], self.r],
                                 name='w',
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.b = self.add_weight(shape=[self.c_num],
                                 name='b',
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=None):
        shp = K.int_shape(inputs)
        si = []
        x = K.reshape(inputs, [-1, shp[-2]*shp[-1]])
        for i in range(self.c_num):
            u1 = self.w[i][:, 0:self.r//2]
            u2 = self.w[i][:, self.r//2:self.r]
            w = K.reshape(K.dot(u1, K.transpose(u1)) - K.dot(u2, K.transpose(u2)), [512*512, 1])
            item = K.dot(x, w) + self.b[i]
            si.append(item)

        outputs = K.softmax(K.concatenate(si, axis=-1), axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list([None, self.c_num])
        return tuple(output_shape)


class CLRBP(Layer):
    def __init__(self,
                 r=4,
                 r2=1,
                 view=4,
                 class_num=10,
                 **kwargs):
        super(CLRBP, self).__init__(**kwargs)
        self.r1 = r
        self.r2 = r2
        self.c_num = class_num
        self.view = view

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=[self.c_num, input_shape[-1] // self.view, self.r1],
                                  name='w1',
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.w2 = self.add_weight(shape=[self.c_num, input_shape[-1], self.r2],
                                  name='w2',
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.l = self.add_weight(shape=[self.c_num],
                                 name='l',
                                 initializer='zeros',
                                 trainable=True)

        self.b = self.add_weight(shape=[self.c_num],
                                 name='b',
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=None):
        shp = K.int_shape(inputs)
        si = []
        x = K.reshape(inputs, [-1, shp[-2]*shp[-1]])

        for i in range(self.c_num):
            if self.r1 <= 0:
                u2 = self.w2[i][:, self.r2 // 2:self.r2]
                v2 = self.w2[i][:, 0:self.r2 // 2]
                w1 = K.reshape(K.dot(u2, K.transpose(u2)) - K.dot(v2, K.transpose(v2)), [shp[-2] * shp[-1], 1])
            else:
                u1 = self.w1[i][:, self.r1 // 2:self.r1]
                v1 = self.w1[i][:, 0:self.r1 // 2]
                w1 = K.dot(u1, K.transpose(u1)) - K.dot(v1, K.transpose(v1))
                w1 = K.reshape(K.tile(w1, [self.view, self.view]), [shp[-2]*shp[-1], 1])

            if self.r2 <= 0:
                u1 = self.w1[i][:, self.r1 // 2:self.r1]
                v1 = self.w1[i][:, 0:self.r1 // 2]
                w1 = K.dot(u1, K.transpose(u1)) - K.dot(v1, K.transpose(v1))
                w2 = K.reshape(K.tile(w1, [self.view, self.view]), [shp[-2]*shp[-1], 1])
            else:
                u2 = self.w2[i][:, self.r2 // 2:self.r2]
                v2 = self.w2[i][:, 0:self.r2 // 2]
                w2 = K.reshape(K.dot(u2, K.transpose(u2)) - K.dot(v2, K.transpose(v2)), [shp[-2] * shp[-1], 1])

            w = w1 * K.sigmoid(self.l[0]) + w2 * (1 - K.sigmoid(self.l[0]))
            item = K.dot(x, w) + self.b[i]
            si.append(item)

        outputs = K.softmax(K.concatenate(si, axis=-1), axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list([None, self.c_num])
        return tuple(output_shape)


