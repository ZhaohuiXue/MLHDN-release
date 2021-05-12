# -*- coding: utf-8 -*-
"""
@author: mengxue.Zhang
"""
from keras.layers.merge import _Merge
import keras.backend as K


# multi-view second-order pooling without flatten
class MVSecondOrderPooling(_Merge):
    def __init__(self,
                 view=4,
                 **kwargs):
        super(MVSecondOrderPooling, self).__init__(**kwargs)
        self.view = view

    def build(self, input_shape):
        self._reshape_required = True

    def _merge_function(self, inputs, training=None):
        if len(inputs) > 1:
            t = K.concatenate(inputs, axis=-1)
        else:
            t = inputs[0]

        output = K.batch_dot(K.permute_dimensions(t, pattern=(0, 2, 1)), t)
        return output

    def compute_output_shape(self, input_shape):
        if self.view == 1:
            output_shape = list([None, input_shape[2], input_shape[2]])
        else:
            output_shape = list([None, input_shape[0][2] * self.view, input_shape[0][2] * self.view])
        return tuple(output_shape)

