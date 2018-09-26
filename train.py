import numpy as np
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras

import DataGenerator
from DataLoader import loadOASISData
from DiffeomorphicRegistrationNet import create_model
from Loggers import Logger
from Preprocessing import readNormalizedVolumeByPath
import random
import multiprocessing as mp

#mp.set_start_method("spawn")
train_config = {
    'batchsize':2,
    'split':0.9,
    'validation':0.1,
    'resolution':(96,96,96),
    'spacings':(1.,1.,1.),
    'epochs': 500,
    'atlas': 'vm',
    'mask_atlas':'yes',
    'model_output': 'model.pkl',
    'conv_filters':[32,16,8,4],
    'encoding_dense': 16
}

training_elements = int(len(loadOASISData())*train_config['split']*(1-train_config['validation']))

data_queue,processes = DataGenerator.stream(3,2,train_config)

validation_data = DataGenerator.getValidationData(train_config)
validation_data_y = DataGenerator.inferYFromBatch(validation_data,train_config)

def train_generator():
    while True:
        minibatch = data_queue.get()
        yield minibatch,DataGenerator.inferYFromBatch(minibatch,train_config)

tb_writers = [
    ("movingImage","image",lambda dic: dic['val_X'][:,:,:,:,1].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("fixedImage","image",lambda dic: dic['val_X'][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("warpedImage","image",lambda dic: dic['pred'][1][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("dispField","image",lambda dic: dic['pred'][0][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("velocityFieldX","image",lambda dic: dic['pred'][2][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("velocityFieldY","image",lambda dic: dic['pred'][2][:,:,:,:,1].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("velocityFieldZ","image",lambda dic: dic['pred'][2][:,:,:,:,2].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("velo","text",lambda dic: "\n".join([ "min: {}\nmax: {}\nX:{}".format(x.min(),x.max(),x) for x in dic['pred'][2][2]])),
    ("disp","text",lambda dic: "\n".join([ "min: {}\nmax: {}\nX:{}".format(x.min(),x.max(),x) for x in dic['pred'][0][2]]))
]

model = create_model(train_config)
tf.set_random_seed(0)
sess = tf.keras.backend.get_session()
with sess.as_default():
    tb = TensorBoard(log_dir='./logs')
    model.fit_generator(generator=train_generator(),
                        validation_data=[validation_data,validation_data_y],
                        epochs=train_config['epochs'],
                        steps_per_epoch=int(training_elements/train_config['batchsize']),
                        callbacks=[tb,Logger(tb_logs=tb_writers)]
                        )
    model.save(train_config['model_output'])

data_queue.close()
for p in processes:
    p.terminate()

