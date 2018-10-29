import functools

import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, Dense, BatchNormalization, Input, Concatenate, UpSampling3D, \
    MaxPool3D, K, Flatten, Reshape, Lambda, LeakyReLU, AveragePooling2D, AveragePooling3D
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

def toDisplacements(steps=7):
    def exponentialMap(args):
        grads = args
        x,y,z = K.int_shape(args)[1:4]

        # ij indexing doesn't change (x,y,z) to (y,x,z)
        grid = tf.expand_dims(tf.stack(tf.meshgrid(
            tf.linspace(0.,x-1.,x),
            tf.linspace(0.,y-1.,y),
            tf.linspace(0.,z-1.,z)
            ,indexing='ij'),-1),
        0)

        # replicate along batch size
        stacked_grids = tf.tile(grid,(tf.shape(grads)[0],1,1,1,1))

        res = tfVectorFieldExp(grads,stacked_grids,n_steps=steps)
        return res
    return exponentialMap

def transformVolume(args):
    x,disp = args
    moving_vol = tf.reshape(x[:,:,:,:,1],(tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3],1))
    transformed_volumes = remap3d(moving_vol,disp)
    return transformed_volumes

def empty_loss(true_y,pred_y):
    return tf.constant(0.,dtype=tf.float32)

def smoothness(batch_size):
    def smoothness_loss(true_y,pred_y):
        print("11 "+str(K.int_shape(pred_y)))
        print("11.1 "+str(K.int_shape(tf.expand_dims(pred_y[:,:,:,:,0],-1))))
        y0 = tf.reshape(pred_y[:,:,:,:,0],[tf.shape(pred_y)[0],*K.int_shape(pred_y)[1:4],1])
        y1 = tf.reshape(pred_y[:,:,:,:,1],[tf.shape(pred_y)[0],*K.int_shape(pred_y)[1:4],1])
        y2 = tf.reshape(pred_y[:,:,:,:,2],[tf.shape(pred_y)[0],*K.int_shape(pred_y)[1:4],1])
        dx = tf.abs(volumeGradients(y0))
        dy = tf.abs(volumeGradients(y1))
        dz = tf.abs(volumeGradients(y2))
        norm = functools.reduce(lambda x,y:x*y,K.int_shape(pred_y)[1:5])*batch_size
        print(norm)
        return tf.reduce_sum((dx+dy+dz)/norm, axis=[1, 2, 3, 4])
    return smoothness_loss

def sampleLoss(true_y,pred_y):
    z_mean = pred_y[:,:,0]
    z_log_sigma = pred_y[:,:,1]
    return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)


def create_model(config):
    input_shape = (*config['resolution'][0:3],2)
    x = Input(shape=input_shape)
    print("1 "+str(K.int_shape(x)))
    # dimension of latent space (batch size by latent dim)
    n_z = config['encoding_dense']

    # encoder
    encoder_filters = config['conv_filters']
    tlayer = x
    for n_filter in encoder_filters:
        tlayer = Conv3D(filters=n_filter, strides=2,kernel_size=3, padding='same',activation='relu')(tlayer)
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm", False)) else tlayer
        print("2 "+str(K.int_shape(tlayer)))

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
    print("3 "+str(K.int_shape(z)))
    print("4 "+str(K.int_shape(init_decoder_tensor)))

    tlayer = Reshape(target_shape=(*lowest_resolution,1))(init_decoder_tensor)
    print("3 "+str(K.int_shape(tlayer)))

    for f,n_filter in list(zip(downsampled_scales,reversed(encoder_filters)))[:-1]:
        conditional_downsampled_moving = AveragePooling3D(pool_size=f,padding='same')(Lambda(lambda arg:tf.expand_dims(arg[:,:,:,:,1],axis=4))(x))
        conditional_stack = Concatenate()([tlayer,conditional_downsampled_moving])
        # upconv
        #tlayer = Conv3D(n_filter,kernel_size=3,activation='relu',padding='same')(UpSampling3D(size=2)(conditional_stack))
        tlayer = Conv3DTranspose(n_filter,kernel_size=3,activation='relu',strides=2,padding='same')(conditional_stack)
        print("5 "+str(K.int_shape(tlayer)))
    tlayer = Conv3D(encoder_filters[-1],kernel_size=3,activation='relu',padding='same')(tlayer)
    print("6 "+str(K.int_shape(tlayer)))
    down_conv = Conv3D(16,kernel_size=5,activation='relu',padding='same')(tlayer)
    print("7 "+str(K.int_shape(down_conv)))
    velocity_maps = Conv3D(3,kernel_size=5,activation='relu',padding='same',name="velocityMap")(down_conv)
    print("8 "+str(K.int_shape(velocity_maps)))

    # TODO: gaussian smoothing

    zLoss = Concatenate(name='zVariationalLoss',axis=2)([
        Lambda(lambda a: tf.expand_dims(a,axis=2))(mu),
        Lambda(lambda a: tf.expand_dims(a,axis=2))(log_sigma)
    ])

    disp = Lambda(toDisplacements(steps=config['exponentialSteps']),name="manifold_walk")(velocity_maps)
    print("9 "+str(K.int_shape(disp)))
    out = Lambda(transformVolume,name="img_warp")([x,disp])
    print("10 "+str(K.int_shape(out)))


    loss = [empty_loss,cc3D(),smoothness(config['batchsize']),sampleLoss]
    lossWeights = [0,1.5,0.00001,0.025]
    model = Model(inputs=x,outputs=[disp,out,velocity_maps,zLoss])
    model.compile(optimizer=Adam(lr=1e-4),loss=loss,loss_weights=lossWeights,metrics=['accuracy'])
    return model
