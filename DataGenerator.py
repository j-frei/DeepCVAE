import os
from multiprocessing import Process, Queue
import numpy as np
from DataLoader import loadOASISData
from Preprocessing import readNormalizedVolumeByPath


def addKeys(config):
    config['batchsize'] = config.get('batchsize',2)
    config['split'] = config.get('split',0.8)
    config['validation'] = config.get('validation',0.1)
    config['resolution'] = config.get('resolution',(128,128,128))
    config['spacings'] = config.get('spacings',(1.5,1.5,1.5))
    return config


def stream(n_elements,n_processes,config):
    config = addKeys(config)
    q = Queue(n_elements)
    pcs = []

    for i in range(n_processes):
        p = Process(target=getBatches,name="Generator{}".format(i+1),args=(q,i,config))
        pcs.append(p)

    for p in pcs:
        p.start()

    return q,pcs


def loadAtlas(config):
    if config.get('atlas','') == "vm":
        atlas_path = os.path.join(os.path.dirname(__file__),"atlas","atlas_norm_resampled.nii.gz")
    else:
        atlas_path = os.path.join(os.path.dirname(__file__),"atlas","icbm_avg_152_t1_tal_lin.nii")
    atlas = readNormalizedVolumeByPath(atlas_path,config)

    if config.get('mask_atlas','') == "yes":
        atlas_mask_path = os.path.join(os.path.dirname(__file__),"atlas","icbm_avg_152_t1_tal_lin_mask.nii")
        atlas_mask = readNormalizedVolumeByPath(atlas_mask_path,config)
        # apply mask
        return (atlas*(atlas_mask>0.5)).astype("float32")
    else:
        return atlas.astype("float32")

def getBatches(*args):
    q,p_number,config = args

    import random
    random.seed(p_number)

    data = loadOASISData()
    train, test = data[:int(len(data) * config['split'])], data[int(len(data) * config['split']):]

    volume_shape = config['resolution']
    atlas = loadAtlas(config)

    data_train = train[int(len(train) * config['validation']):]

    while True:
        minibatch = np.empty(shape=(config['batchsize'],*volume_shape,2))

        for i in range(config['batchsize']):
            idx_volume = random.choice(list(range(len(data_train))))
            vol = readNormalizedVolumeByPath( data_train[idx_volume]['img'], config )
            minibatch[i,:,:,:,0] = atlas.reshape(volume_shape).astype("float32")
            minibatch[i,:,:,:,1] = vol.reshape(volume_shape).astype("float32")

        q.put(minibatch)

def getValidationData(config):
    data = loadOASISData()
    train, test = data[:int(len(data) * config['split'])], data[int(len(data) * config['split']):]
    volume_shape = config['resolution']
    atlas = loadAtlas(config)

    data_val = train[:int(len(train) * config['validation'])]
    l = len(data_val)
    val = np.empty(shape=(l,*volume_shape,2))

    for i in range(l):
        vol = readNormalizedVolumeByPath( data_val[i]['img'] ,config )
        val[i,:,:,:,0] = atlas.reshape(volume_shape).astype("float32")
        val[i,:,:,:,1] = vol.reshape(volume_shape).astype("float32")
    
    return val


def getTestData(config):
    data = loadOASISData()
    data_test = data[int(len(data) * config['split']):]
    volume_shape = config['resolution']
    atlas = loadAtlas(config)

    l = len(data_test)
    test = np.empty(shape=(l, *volume_shape, 2))

    for i in range(l):
        vol = readNormalizedVolumeByPath(data_test[i]['img'] ,config )
        test[i, :, :, :, 0] = atlas.reshape(volume_shape).astype("float32")
        test[i, :, :, :, 1] = vol.reshape(volume_shape).astype("float32")

    return test

def inferYFromBatch(batch,config):
    y = [
        np.asarray([ np.zeros(shape=(*config['resolution'], 3)).astype(np.float32) for _ in range(len(batch)) ]),
        np.asarray([ volumes[:,:,:,0].reshape(*config['resolution'], 1) for volumes in batch ]),
        np.asarray([np.zeros(shape=(*config['resolution'], 2)).astype(np.float32) for _ in range(len(batch))]),
        np.asarray([np.zeros(shape=(config['encoding_dense'], 2)).astype(np.float32) for _ in range(len(batch))]),
    #    np.asarray([ [[0., 0., 0.] for _ in range(20)] for _ in range(len(batch))]).reshape(len(batch), 20, 3),
    #    np.asarray([ [0. for _ in range(20)] for _ in range(len(batch))]).reshape(len(batch), 20, 1),
    #    np.asarray([ [0. for _ in range(20)] for _ in range(len(batch))]).reshape(len(batch), 20, 1),
    ]
    return y
