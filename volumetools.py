from tensorflow.python.ops import array_ops
import tensorflow as tf
import numpy as np
from keras.layers import K
from dense_3D_spatial_transformer import Dense3DSpatialTransformer


def volumeGradients(tf_vf):
    # batch_size, xaxis, yaxis, zaxis, depth = \
    shapes = tf.shape(tf_vf)
    # batch_size = tf.placeholder("float",shape=None)
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

    '''
    normalization = lambda unnormalized_tensor: tf.div(
        tf.subtract(
            unnormalized_tensor,
            tf.reduce_min(unnormalized_tensor)
        ),
        tf.subtract(
            tf.reduce_max(unnormalized_tensor),
            tf.reduce_min(unnormalized_tensor)
        )
    )
    
    new_shape = np.array([-1, xaxis, yaxis, zaxis, 3], dtype=np.float32)
    return tf.map_fn(normalization, tf.reshape(array_ops.stack([dx, dy, dz], 4), new_shape))
    '''
    return tf.reshape(array_ops.stack([dx, dy, dz], 4), [shapes[0], shapes[1], shapes[2], shapes[3], 3])


def remap3d(tf_in_vol, tf_offsets):
    ox = tf_offsets[:, :, :, :, 1]
    oy = tf_offsets[:, :, :, :, 0]
    oz = tf_offsets[:, :, :, :, 2]

    offsets_swap = tf.squeeze(
        tf.stack([tf.expand_dims(ox, -1), tf.expand_dims(oy, -1), tf.expand_dims(oz, -1)], 4),
        -1
    )

    return Dense3DSpatialTransformer()([tf_in_vol, offsets_swap])


def tfVectorFieldExp(grad, grid):
    N = 4
    shapes = tf.shape(grad)
    batch_size, size_x, size_y, size_z, channels = shapes[0], shapes[1], shapes[2], shapes[3], shapes[4]

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

        dvx = remap3d(dvx, cache_tf)
        dvy = remap3d(dvy, cache_tf)
        dvz = remap3d(dvz, cache_tf)

    ox = dvx - id_x
    oy = dvy - id_y
    oz = dvz - id_z
    out = tf.reshape(tf.stack([ox, oy, oz], 4), [batch_size, size_x, size_y, size_z, 3])
    return out
