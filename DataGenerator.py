import os
from multiprocessing import Process, Queue
import numpy as np
from DataLoader import loadOASISData
from Preprocessing import readNormalizedVolumeByPath, readNormalizedAtlasAndITKAtlas
import logging

def stream(n_elements,n_processes,config):
    q = Queue(n_elements)
    pcs = []

    for i in range(n_processes):
        p = Process(target=getBatches,name="Generator{}".format(i+1),args=(q,i,config))
        pcs.append(p)

    for p in pcs:
        p.start()

    return q,pcs


def loadAtlas(config):
    atlas_path = os.path.join(os.path.dirname(__file__),"atlas",config.get('atlas','atlas.nii.gz'))
    atlas,itk_atlas = readNormalizedAtlasAndITKAtlas(atlas_path)
    if not config.get('resolution') or not config.get('spacing'):
        config['resolution'] = np.asarray(itk_atlas.GetSize())
        config['spacing'] = np.asarray(itk_atlas.GetSpacing())
        logging.info("Setting resolution and spacing according to atlas:\nresolution: {}\nspacing: {}".format(config['resolution'],config['spacing']))
    return atlas.astype("float32"), itk_atlas

def getBatches(*args):
    q,p_number,config = args

    import random
    random.seed(p_number)
    atlas, itk_atlas = loadAtlas(config)

    data = loadOASISData()
    train, test = data[:int(len(data) * config['split'])], data[int(len(data) * config['split']):]

    volume_shape = config['resolution']

    data_train = train[int(len(train) * config['validation']):]

    while True:
        minibatch = np.empty(shape=(config['batchsize'],*volume_shape,2))

        for i in range(config['batchsize']):
            idx_volume = random.choice(list(range(len(data_train))))
            vol = readNormalizedVolumeByPath( data_train[idx_volume]['img'], itk_atlas )
            minibatch[i,:,:,:,0] = atlas.reshape(volume_shape).astype("float32")
            minibatch[i,:,:,:,1] = vol.reshape(volume_shape).astype("float32")

        q.put(minibatch)

def getValidationData(config):
    atlas, itk_atlas = loadAtlas(config)
    data = loadOASISData()
    train, test = data[:int(len(data) * config['split'])], data[int(len(data) * config['split']):]
    volume_shape = config['resolution']

    data_val = train[:int(len(train) * config['validation'])]
    l = len(data_val)
    val = np.empty(shape=(l,*volume_shape,2))

    for i in range(l):
        vol = readNormalizedVolumeByPath( data_val[i]['img'], itk_atlas )
        val[i,:,:,:,0] = atlas.reshape(volume_shape).astype("float32")
        val[i,:,:,:,1] = vol.reshape(volume_shape).astype("float32")
    
    return val


def getTestData(config):
    atlas, itk_atlas = loadAtlas(config)
    data = loadOASISData()
    data_test = data[int(len(data) * config['split']):]
    volume_shape = config['resolution']

    l = len(data_test)
    test = np.empty(shape=(l, *volume_shape, 2))

    for i in range(l):
        vol = readNormalizedVolumeByPath(data_test[i]['img'], itk_atlas )
        test[i, :, :, :, 0] = atlas.reshape(volume_shape).astype("float32")
        test[i, :, :, :, 1] = vol.reshape(volume_shape).astype("float32")

    return test

def inferYFromBatch(batch,config):
    velo_res = config['resolution']
    if config['half_res']:
        velo_res = [ int(axis / 2) for axis in velo_res]
    y = [
        # displacement field at full resolution
        np.asarray([ np.zeros(shape=(*config['resolution'], 3)).astype(np.float32) for _ in range(len(batch)) ]),
        # warped moving image
        np.asarray([ volumes[:,:,:,0].reshape(*config['resolution'], 1) for volumes in batch ]),
        np.asarray([np.zeros(shape=(*velo_res, 3)).astype(np.float32) for _ in range(len(batch))]),
        np.asarray([np.zeros(shape=(config['encoding_dense'], 2)).astype(np.float32) for _ in range(len(batch))]),
    ]
    return y
