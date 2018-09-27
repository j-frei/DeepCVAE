import functools

import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, Dense, BatchNormalization, Input, Concatenate, UpSampling3D, \
    MaxPool3D, K, Flatten, Reshape, Lambda, LeakyReLU, AveragePooling2D, AveragePooling3D
from tensorflow.contrib.distributions import MultivariateNormalDiag as MultivariateNormal
from losses import cc3D
from volumetools import volumeGradients, tfVectorFieldExp, remap3d

def sampling(args):
    z_mean = args[0]
    z_log_sigma = args[1]
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    #flattened_dim = functools.reduce(lambda x,y:x*y,[*dim,3])
    epsilon = tf.reshape(K.random_normal(shape=(batch, dim),dtype=tf.float32),(batch,dim))
    xout = z_mean + K.exp(z_log_sigma) * epsilon
    return xout

def _meshgrid(height, width, depth):
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                                            tf.cast(width, tf.float32)-1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                               tf.cast(height, tf.float32)-1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))

    x_t = tf.tile(tf.expand_dims(x_t, 2), [1, 1, depth])
    y_t = tf.tile(tf.expand_dims(y_t, 2), [1, 1, depth])

    z_t = tf.linspace(0.0, tf.cast(depth, tf.float32)-1.0, depth)
    z_t = tf.expand_dims(tf.expand_dims(z_t, 0), 0)
    z_t = tf.tile(z_t, [height, width, 1])

    return x_t, y_t, z_t

def toDisplacements(n_squaringScaling):
    def displacementWalk(args):
        grads = args
        height = K.shape(grads)[1]
        width = K.shape(grads)[2]
        depth = K.shape(grads)[3]

        _grid = tf.reshape(tf.stack(_meshgrid(height,width,depth),-1),(1,height,width,depth,3))
        _stacked = tf.tile(_grid,(tf.shape(grads)[0],1,1,1,1))
        grids = tf.reshape(_stacked,(tf.shape(grads)[0],tf.shape(grads)[1],tf.shape(grads)[2],tf.shape(grads)[3],3))

        out = grads
        for i in range(n_squaringScaling):
            #out = tfVectorFieldExp(out,grids)
            out = out + tfVectorFieldExp(out,grids)
        return out
    return displacementWalk


def transformVolume(args):
    x,disp = args
    moving_vol = tf.reshape(x[:,:,:,:,1],(tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3],1))
    #transformed_volumes = Dense3DSpatialTransformer()([moving_vol,disp])
    transformed_volumes = remap3d(moving_vol,disp)
    return transformed_volumes

def empty_loss(true_y,pred_y):
    return tf.constant(0.,dtype=tf.float32)

def smoothness_loss(true_y,pred_y):
    dx = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,0],-1)))
    dy = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,1],-1)))
    dz = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,2],-1)))
    return 1e-5*tf.reduce_sum(dx+dy+dz, axis=[1, 2, 3, 4])

def sampleLoss(true_y,pred_y):
    z_mean = pred_y[:,:,0]
    z_log_sigma = pred_y[:,:,1]
    return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)


def create_model(config):
    input_shape = (*config['resolution'][0:3],2)

    x = Input(shape=input_shape)
    # dimension of latent space (batch size by latent dim)
    n_z = config['encoding_dense']

    # encoder
    encoder_filters = config['conv_filters']
    tlayer = x
    for n_filter in encoder_filters:
        tlayer = Conv3D(filters=n_filter, strides=2,kernel_size=3, padding='same',activation='relu')(tlayer)
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm", False)) else tlayer

    tlayer = Flatten()(tlayer)
    # dense ReLU layer to mu and sigma
    mu = Dense(n_z,activation='linear')(tlayer)
    log_sigma = Dense(n_z,activation='linear')(tlayer)
    # sampled latent space
    z = Lambda(sampling)([mu,log_sigma])

    # decoder
    downsampled_scales = list(reversed([ 2**i for i,_ in enumerate(encoder_filters)]))
    lowest_resolution = [ int( l / downsampled_scales[0]) for l in config['resolution'][0:3] ]
    lowest_dim = functools.reduce(lambda x,y: x*y,lowest_resolution)

    init_decoder_tensor = Dense(lowest_dim,activation='relu')(z)

    tlayer = Reshape(target_shape=(*lowest_resolution,1))(init_decoder_tensor)

    for f,n_filter in list(zip(downsampled_scales,reversed(encoder_filters)))[:-1]:
        conditional_downsampled_moving = AveragePooling3D(pool_size=f,padding='same')(Lambda(lambda arg:tf.expand_dims(arg[:,:,:,:,1],axis=4))(x))
        conditional_stack = Concatenate()([tlayer,conditional_downsampled_moving])
        # upconv
        #tlayer = Conv3D(n_filter,kernel_size=3,activation='relu',padding='same')(UpSampling3D(size=2)(conditional_stack))
        tlayer = Conv3DTranspose(n_filter,kernel_size=3,activation='relu',strides=2,padding='same')(conditional_stack)
    tlayer = Conv3D(encoder_filters[-1],kernel_size=3,activation='relu',padding='same')(tlayer)
    down_conv = Conv3D(16,kernel_size=5,activation='relu',padding='same')(tlayer)
    velocity_maps = Conv3D(3,kernel_size=5,activation='relu',padding='same',name="velocityMap")(down_conv)

    # TODO: gaussian smoothing

    zLoss = Concatenate(name='zVariationalLoss',axis=2)([
        Lambda(lambda a: tf.expand_dims(a,axis=2))(mu),
        Lambda(lambda a: tf.expand_dims(a,axis=2))(log_sigma)
    ])

    disp = Lambda(toDisplacements(n_squaringScaling=1),name="manifold_walk")(velocity_maps)
    out = Lambda(transformVolume,name="img_warp")([x,disp])


    loss = [empty_loss,cc3D(),smoothness_loss,sampleLoss]
    lossWeights = [0,1.5,0.00001,0.025]
    model = Model(inputs=x,outputs=[disp,out,velocity_maps,zLoss])
    model.compile(optimizer=Adam(lr=1e-4),loss=loss,loss_weights=lossWeights,metrics=['accuracy'])
    return model
