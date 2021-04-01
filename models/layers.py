import tensorflow as tf


def maxPool_layer(x, poolSize):
    # x -> [batch,H,W,Channels]
    return tf.nn.max_pool2d(input=x, ksize=[1, poolSize, poolSize, 1], strides=[1, poolSize, poolSize, 1],
                            padding="SAME")


def conv_layer(input_x, w, b):
    # input_x -> [batch,H,W,Channels]
    # filter_shape -> [filters H, filters W, Channels In, Channels Out]
    y = tf.nn.conv2d(input=input_x, filters=w, strides=[1, 1, 1, 1], padding='SAME') + b
    y = tf.nn.relu(y)
    return y


def fullyConnected_layer(input_layer, w, b):
    y = tf.add(tf.matmul(input_layer, w), b)
    return y


def get_tfVariable(shape, name):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name, trainable=True, dtype=tf.float32)