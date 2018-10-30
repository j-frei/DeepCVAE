import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import keras
import logging, os
import DataGenerator
from DataLoader import loadOASISData
from DiffeomorphicRegistrationNet import create_model
from Loggers import Logger
from Preprocessing import readNormalizedVolumeByPath
import random
import multiprocessing as mp

#mp.set_start_method("spawn")
train_config = {
    'batchsize':1,
    'split':0.9,
    'validation':0.1,
    'half_res':True,
    'epochs': 200,
    'atlas': 'atlas.nii.gz',
    'model_output': 'model.pkl',
    'exponentialSteps': 7,
    'conv_filters':[32,16,8],
    'encoding_dense': 16
}

training_elements = int(len(loadOASISData())*train_config['split']*(1-train_config['validation']))

data_queue,processes = DataGenerator.stream(2,1,train_config)

validation_data = DataGenerator.getValidationData(train_config)
validation_data_y = DataGenerator.inferYFromBatch(validation_data,train_config)

def train_generator():
    while True:
        minibatch = data_queue.get()
        yield minibatch,DataGenerator.inferYFromBatch(minibatch,train_config)


velo_res = np.array(train_config['resolution'],dtype=np.int)
if train_config['half_res']:
    velo_res = (velo_res / 2).astype(np.int)

tb_writers = [
    ("movingImage","image",lambda dic: dic['val_X'][:,:,:,:,1].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("fixedImage","image",lambda dic: dic['val_X'][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("warpedImage","image",lambda dic: dic['pred'][1][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("dispField","image",lambda dic: dic['pred'][0][:,:,:,:,0].reshape(dic['batchsize'],*train_config['resolution'])[:,:,:,int(train_config['resolution'][2]/2.)].astype("float32").reshape(dic['batchsize'],*train_config['resolution'][:2],1)),
    ("velocityFieldX","image",lambda dic: dic['pred'][2][:,:,:,:,0].reshape(dic['batchsize'],*velo_res)[:,:,:,int(velo_res[2]/2.)].astype("float32").reshape(dic['batchsize'],*velo_res[:2],1)),
    ("velocityFieldY","image",lambda dic: dic['pred'][2][:,:,:,:,1].reshape(dic['batchsize'],*velo_res)[:,:,:,int(velo_res[2]/2.)].astype("float32").reshape(dic['batchsize'],*velo_res[:2],1)),
    ("velocityFieldZ","image",lambda dic: dic['pred'][2][:,:,:,:,2].reshape(dic['batchsize'],*velo_res)[:,:,:,int(velo_res[2]/2.)].astype("float32").reshape(dic['batchsize'],*velo_res[:2],1)),
    ("velo","text",lambda dic: "\n".join([ "min: {}\nmax: {}\nX:{}".format(x.min(),x.max(),x) for x in dic['pred'][2][2]])),
    ("disp","text",lambda dic: "\n".join([ "min: {}\nmax: {}\nX:{}".format(x.min(),x.max(),x) for x in dic['pred'][0][2]]))
]

cp_models_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"models","weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5")

model = create_model(train_config)
tf.set_random_seed(0)
sess = tf.keras.backend.get_session()
with sess.as_default():
    tb = TensorBoard(log_dir='./logs')
    logging.info("Start Training")
    model.fit_generator(generator=train_generator(),
                        validation_data=[validation_data,validation_data_y],
                        epochs=train_config['epochs'],
                        steps_per_epoch=int(training_elements/train_config['batchsize']),
                        callbacks=[tb,Logger(tb_logs=tb_writers),ModelCheckpoint(cp_models_path)]
                        )
    model.save(train_config['model_output'])

data_queue.close()
for p in processes:
    p.terminate()

