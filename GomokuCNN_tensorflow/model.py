import tensorflow as tf


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def con2d(activation_map, weight):
    return tf.nn.conv2d(activation_map, weight, strides=[1, 3, 3, 1], padding='SAME')


def max_pool_2x2(activation_map):
    return tf.nn.max_pool(activation_map, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')


sess = tf.Session()

x = tf.placeholder(tf.int32, shape=[None, 15, 15])
y_ = tf.placeholder(tf.int32, shape=[None, 2, 10])

W_conv1 = tf.Variable(tf.zeros([784, 10]))
b_conv2 = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

