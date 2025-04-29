import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, models, layers, losses, metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime


# ------------------------
# Channel Attention Layer
# ------------------------

class ChannelAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ch_attention = tf.Variable(tf.ones((1, 1, 1, 1, 4), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return inputs * self.ch_attention

#Load data
T = dict(np.load('/path/patched_data.npz'))
V=T
#Load models
folder =  '/models_path/'

model = models.load_model(folder + 'final_training.keras', compile = True,
                          custom_objects={'ChannelAttention': ChannelAttention}, safe_mode=False)

backbone = models.load_model(folder + 'final_backbone.keras', compile = True,
                          custom_objects={'ChannelAttention': ChannelAttention}, safe_mode=False)


def generator(my_data, batch_size):
    while True:

        q =  tf.random.uniform(shape=[batch_size], maxval=my_data['dat'].shape[0]-1, dtype=tf.int32)

        dat = my_data['dat'][q,...].astype('float32')
        lbl = my_data['lbl'][q,...].astype('float32')

        yield dat, lbl

gen_tra = generator(T, 5)
gen_val = generator(V, 5)


x = next(gen_tra)

q = model.predict(x[0])

Q = {'input': x[0], 'output': q, 'ground_truth': x[1]}


np.savez('output_sample.npz', **Q)
