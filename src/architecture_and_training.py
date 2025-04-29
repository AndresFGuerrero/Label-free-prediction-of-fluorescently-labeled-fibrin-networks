import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, models, layers, losses, metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime

# ------------------------
# Normalization Utilities
# ------------------------

def Z_norm_tf(x):
    """Standard z-score normalization for tensors."""
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)

def minmax_tf(x):
    """Min-max normalization for tensors."""
    return (x - tf.math.reduce_min(x)) / (tf.math.reduce_max(x) - tf.math.reduce_min(x))

# ------------------------
# Custom Layers
# ------------------------

class LayerAbs(layers.Layer):
    def call(self, x):
        return tf.math.abs(x)

class LayerMean(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x)

class LayerMinMax(layers.Layer):
    def call(self, x):
        return minmax_tf(x)

class LayerZnorm(layers.Layer):
    def call(self, x):
        return Z_norm_tf(x)

class LayerSSIM(layers.Layer):
    def call(self, y_true, y_pred):
        return tf.image.ssim(
            LayerMinMax()(y_true),
            LayerMinMax()(y_pred),
            max_val=1,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
        )

# ------------------------
# Custom Loss
# ------------------------

class GeneratorLoss(losses.Loss):
    def call(self, y_true, y_pred):
        mae = LayerAbs()(y_pred - y_true)
        mqe = LayerMean()(mae**3)
        mae = LayerMean()(mae)
        ssim = LayerSSIM()(y_true, y_pred)
        return LayerMean()(mae + 0.1 * mqe + 0.0001 * (1 - ssim))

# ------------------------
# Channel Attention Layer
# ------------------------

class ChannelAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ch_attention = tf.Variable(tf.ones((1, 1, 1, 1, 4), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return inputs * self.ch_attention

# ------------------------
# U-Net 3D Backbone
# ------------------------

def unet_3D(input_shape):

    _,Z,Y,X,C = input_shape

    x = Input(shape=(None, Y, X, C), dtype='float32')


    # --- Define kwargs dictionary
    kwargs2 = {
        'kernel_size': (1, 5, 5),
        'padding': 'same'}


     # --- Define kwargs dictionary
    kwargs3 = {
        'kernel_size': (3, 5, 5),
        'padding': 'same'}

    # --- Define lambda functions
    conv2D = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs2)(x)
    conv3D = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs3)(x)
    norm = lambda x : layers.BatchNormalization()(x)

    relu = lambda x : layers.ReLU()(x)

    # --- Define single transpose
    tran2D = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs2)(x)
    tran3D = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs3)(x)

    aver = tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=1, padding='same')

    # --- Define transpose block
    tran2_2D = lambda filters, x : relu(norm(aver(tran2D(x, filters, strides=(1, 2, 2)))))
    # --- Define transpose block
    tran2_3D = lambda filters, x : relu(norm(aver(tran3D(x, filters, strides=(1, 2, 2)))))

    # --- Define stride-1, stride-2 blocks
    conv1_2D = lambda filters, x : relu(norm(conv2D(x, filters, strides=1)))
    conv1_3D = lambda filters, x : relu(norm(conv3D(x, filters, strides=1)))
    conv2_2D = lambda filters, x : relu(norm(conv2D(x, filters, strides=(1, 2, 2))))
    conv2_3D = lambda filters, x : relu(norm(conv3D(x, filters, strides=(1, 2, 2))))
    # --- Concatenate
    concat = lambda a, b : layers.Concatenate()([a, b])
    # --- Define model layers
    r0 = 8
    r1 = 2*r0#16
    r2 = 2*r1#32
    r3 = 2*r2#64
    r4 = 2*r3#128


    num_channels = x.shape[-1]

    # Example usage
    ch_attention_layer = ChannelAttention()

    l0 = ch_attention_layer(x)


    # --- Define contracting layers
    l1 = conv1_3D(r0, l0)
    l2 = conv1_3D(r1, conv2_3D(r1, l1))
    l3 = conv1_3D(r2, conv2_3D(r2, l2))
    l4 = conv1_3D(r3, conv2_3D(r3, l3))
    l5 = conv1_3D(r4, conv2_3D(r4, l4))
    # --- Define expanding layers
    l6 = conv1_3D(r3, tran2_3D(r3, l5))
    # --- Define expanding layers
    l7  = conv1_3D(r2,tran2_3D(r2, concat(l4, l6)))
    l8  = conv1_3D(r1,tran2_3D(r1, concat(l3, l7)))
    l9  = conv1_3D(r0,tran2_3D(r0,  concat(l2, l8)))
    l_h1_10 = conv1_3D(r0,conv1_3D(r0,  concat(l1, l9)))

    # --- Create logits
    logits = layers.Conv3D(filters=1, **kwargs2)(l_h1_10)

    dic_out = {'l1':l1, 'l2':l2, 'l3':l3, 'l4':l4, 'l5':l5,
               'l6':l6, 'l7':l7, 'l8':l8, 'l9':l9,
               'output': logits,}


    # --- Create model
    backbone = Model(inputs=x, outputs=dic_out)

    return backbone


# ------------------------
# Model Creation Wrappers
# ------------------------

def create_wrapper(input_shape, batch=1):
    c_d = input_shape[-1]
    c_l = 1

    dat_input = Input(shape=(None, None, None, c_d), dtype='float32', name='dat')
    unet = unet_3D(dat_input.shape)
    y_pred = unet(dat_input)['output']

    model = Model(inputs=dat_input, outputs=y_pred)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=lambda y_true, y_pred: GeneratorLoss()(y_true, y_pred),
        metrics=[
            metrics.MeanAbsoluteError(name='mae'),
            metrics.MeanSquaredError(name='mse'),
        ]
    )

    return model, unet

# ------------------------
# Training Function
# ------------------------

def Train(input_shape, data, valid, name_of_run, epochs, batch_size=1):
    training, backbone = create_wrapper(input_shape)

    callbacks = [
        ModelCheckpoint('best_training_tra.keras', monitor='mae', save_best_only=True, mode='min', verbose=1),
        ModelCheckpoint('best_training_val.keras', monitor='val_mae', save_best_only=True, mode='min', verbose=1),
        ModelCheckpoint('best_weights.weights.h5', monitor='mae', save_best_only=True, save_weights_only=True, mode='min', verbose=1),
        TensorBoard(log_dir=f"logs2/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name_of_run}", histogram_freq=1),
    ]

    history = training.fit(
        x=data,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=valid,
        validation_steps=10,
        validation_freq=1,
        shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return training, backbone, history

# ------------------------
# Data Generator
# ------------------------

def generator(my_data, batch_size):
    while True:
        idx = tf.random.uniform(shape=[batch_size], maxval=my_data['dat'].shape[0]-1, dtype=tf.int32)
        dat = my_data['dat'][idx, ...].astype('float32')
        lbl = my_data['lbl'][idx, ...].astype('float32')
        yield dat, lbl

T = dict(np.load('/path/data_patched.npz'))

# NOTE:
# In a proper experimental setup, training and validation datasets must be **independent** 
# to ensure unbiased model evaluation and to prevent data leakage.
#
# Here, for demonstration purposes only, we assign the same dataset `T` to both 
# training (`T`) and validation (`V`) cohorts. 
# 
# IMPORTANT: 
# Before actual training or serious deployment, 
# make sure to **split your data appropriately** into distinct training and validation sets

#
V = T  # Validation = Training (for code testing only; NOT recommended for real experiments)


gen_tra = generator(T, batch_size=5)
gen_val = generator(V, batch_size=5)



name_of_run = 'all_channels_experiment'
training, backbone, history = Train((20, 256, 256, 4), gen_tra, gen_val, name_of_run, epochs=2000, batch_size=5)



training.save('./final_training.keras')
backbone.save('./final_backbone.keras')

