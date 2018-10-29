from tensorflow.python.ops import array_ops
import tensorflow as tf
import numpy as np
from keras.layers import K
from dense_3D_spatial_transformer import Dense3DSpatialTransformer

def remap3d(tf_in_vol, tf_offsets):
    # apply dense 3d interpolation with reordered axes (x,y,z) -> (y,x,z) of trf tensor
    transformed = Dense3DSpatialTransformer()([
                                       tf.transpose(tf_in_vol,perm=[0,2,1,3,4]),
                                        tf.transpose(tf_offsets,perm=[0,2,1,3,4])])
    return tf.transpose(transformed,perm=[0,2,1,3,4])

def tfVectorFieldExp(grad, grid,n_steps):
    N = n_steps

    shapes = tf.shape(grad)
    batch_size, size_x, size_y, size_z, channels = shapes[0], *K.int_shape(grad)[1:5]

    id_x = tf.reshape(grid[:, :, :, :, 0], [batch_size, size_x, size_y, size_z, 1])
    id_y = tf.reshape(grid[:, :, :, :, 1], [batch_size, size_x, size_y, size_z, 1])
    id_z = tf.reshape(grid[:, :, :, :, 2], [batch_size, size_x, size_y, size_z, 1])

    ux = grad[:, :, :, :, 0]
    uy = grad[:, :, :, :, 1]
    uz = grad[:, :, :, :, 2]

    dvx = ux / (2.0 ** N)
    dvy = uy / (2.0 ** N)
    dvz = uz / (2.0 ** N)
    dvx = id_x + tf.reshape(dvx, [batch_size, size_x, size_y, size_z, 1])
    dvy = id_y + tf.reshape(dvy, [batch_size, size_x, size_y, size_z, 1])
    dvz = id_z + tf.reshape(dvz, [batch_size, size_x, size_y, size_z, 1])

    for n in range(0, N - 1):
        cache_tf = tf.stack([dvx - id_x, dvy - id_y, dvz - id_z], 4)
        cache_tf = tf.reshape(cache_tf, [batch_size, size_x, size_y, size_z, 3])

        dvx = remap3d(dvx, cache_tf) + tf.expand_dims(cache_tf[:,:,:,:,0],-1)
        dvy = remap3d(dvy, cache_tf) + tf.expand_dims(cache_tf[:,:,:,:,1],-1)
        dvz = remap3d(dvz, cache_tf) + tf.expand_dims(cache_tf[:,:,:,:,2],-1)

    ox = dvx - id_x
    oy = dvy - id_y
    oz = dvz - id_z
    out = tf.reshape(tf.stack([ox, oy, oz], 4), [batch_size, size_x, size_y, size_z, 3])
    return out

def upsample(tf_in_vol):
    batch_size = tf.shape(tf_in_vol)[0]
    x,y,z = tf_in_vol.get_shape().as_list()[1:4]

    xs_low = -1.*tf.linspace(0.,x-1.,x) / 2. - 0.50
    ys_low = -1.*tf.linspace(0.,y-1.,y) / 2. - 0.50
    # Why is the z-axis not shifted by 0.5?!
    zs_low = -1.*tf.linspace(0.,z-1.,z) / 2. #+ 0.50
    xs_high = xs_low + (x/2.)
    ys_high = ys_low + (y/2.)
    zs_high = zs_low + (z/2.)
    xdic = {'l':xs_low,'h':xs_high}
    ydic = {'l':ys_low,'h':ys_high}
    zdic = {'l':zs_low,'h':zs_high}

    def to_trf(setup):
        # 'ij' returns xx,yy,zz whereas 'xy' returns yy,xx,zz.
        # Probably, tf assumes yy,xx,zz input ordering.
        xx,yy,zz = tf.meshgrid(xdic[setup[0]],ydic[setup[1]],zdic[setup[2]],indexing='ij')
        trf = tf.tile(tf.expand_dims(tf.stack([xx,yy,zz],3),0),[batch_size,1,1,1,1])
        return trf

    # 8x mini cubes
    lll = remap3d(tf_in_vol,to_trf("lll"))
    lhl = remap3d(tf_in_vol,to_trf("lhl"))
    llh = remap3d(tf_in_vol,to_trf("llh"))
    lhh = remap3d(tf_in_vol,to_trf("lhh"))
    hll = remap3d(tf_in_vol,to_trf("hll"))
    hhl = remap3d(tf_in_vol,to_trf("hhl"))
    hlh = remap3d(tf_in_vol,to_trf("hlh"))
    hhh = remap3d(tf_in_vol,to_trf("hhh"))

    # 4x quaders along x axis
    ll_lowz = tf.concat([lll,hll],axis=1)
    hh_lowz = tf.concat([lhl,hhl],axis=1)
    ll_highz = tf.concat([llh,hlh],axis=1)
    hh_highz = tf.concat([lhh,hhh],axis=1)

    # 2x quaders along y axis
    low_z = tf.concat([ll_lowz,hh_lowz],axis=2)
    high_z = tf.concat([ll_highz,hh_highz],axis=2)

    # 1x final quader
    result = tf.concat([low_z,high_z],axis=3)
    result = tf.reshape(result,[batch_size,x*2,y*2,z*2,1])

    return result

def volumeGradients(tf_vf):
    # batch_size, xaxis, yaxis, zaxis, depth = \
    shapes = (tf.shape(tf_vf)[0],*K.int_shape(tf_vf)[1:])
    dx = tf_vf[:, 1:, :, :, :] - tf_vf[:, :-1, :, :, :]
    dy = tf_vf[:, :, 1:, :, :] - tf_vf[:, :, :-1, :, :]
    dz = tf_vf[:, :, :, 1:, :] - tf_vf[:, :, :, :-1, :]

    # Return tensors with same size as original image by concatenating
    # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    shape = tf.stack([shapes[0], 1, shapes[2], shapes[3], shapes[4]])
    dx = array_ops.concat([dx, tf.zeros(shape, tf_vf.dtype)], 1)
    dx = array_ops.reshape(dx, tf.shape(tf_vf))

    # shape = tf.stack([batch_size, xaxis, 1, zaxis, depth])
    shape = tf.stack([shapes[0], shapes[1], 1, shapes[3], shapes[4]])
    dy = array_ops.concat([dy, array_ops.zeros(shape, tf_vf.dtype)], 2)
    dy = array_ops.reshape(dy, tf.shape(tf_vf))

    # shape = tf.stack([batch_size, xaxis, yaxis, 1, depth])
    shape = tf.stack([shapes[0], shapes[1], shapes[2], 1, shapes[4]])
    dz = array_ops.concat([dz, array_ops.zeros(shape, tf_vf.dtype)], 3)
    dz = array_ops.reshape(dz, tf.shape(tf_vf))

    return tf.reshape(array_ops.stack([dx, dy, dz], 4), [shapes[0], shapes[1], shapes[2], shapes[3], 3])
