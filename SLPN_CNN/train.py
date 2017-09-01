import tensorflow as tf
import numpy as np
import os
import time

from ..util.DatasetManager.renjunet import Record
from .model import PolicyNetwork


def load_dataset(num_traingset=1400000):
    x_boards = Record.load_record_input()
    y_labels = Record.load_record_output(record_decoding_option='sequence')
    x_train, x_test = x_boards[:num_traingset], x_boards[num_traingset:]
    y_train, y_test = y_labels[:num_traingset], y_labels[num_traingset:]
    return (x_train, x_test), (y_train, y_test)


def train(name_policy_network, start_epoch, end_epoch, batch_size, learning_rate):
    # load data
    (x_train, x_test), (y_train, y_test) = load_dataset()
    # build the network
    state = tf.placeholder(tf.int8, shape=[None, 15, 15, 2])
    action = tf.placeholder(tf.int8, shape=[None, 225])
    keep_prob = tf.placeholder(tf.float32)
    pn = PolicyNetwork({'state': state,'action': action,'keep_prob': keep_prob}, name=name_policy_network, learning_rate=learning_rate)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if start_epoch >= 2:
            pn.restore(sess, start_epoch - 1)
        for epoch in range(start_epoch, end_epoch + 1):
            xy_train = list(zip(x_train, y_train))
            np.random.shuffle(xy_train)
            x_train, y_train = zip(*xy_train)
            start_time = time.time()
            for i in range(28000):
                batch_start = i * batch_size
                feed_dict={state: x_train[batch_start:batch_start + batch_size], action: y_train[batch_start:batch_start + batch_size], keep_prob: 0.5}
                sess.run(pn.train_op, feed_dict)
                if (i + 1) % 2000 is 0:
                    feed_dict={state: x_train[batch_start:batch_start + batch_size], action: y_train[batch_start:batch_start + batch_size], keep_prob: 1.0}
                    train_accuracy, loss = sess.run([pn.accuracy, pn.loss], feed_dict)
                    end_time = time.time()
                    print('step %d, training accuracy %g, loss %g, time consumption %gs' % (i + 1, train_accuracy, loss, end_time - start_time))
                    start_time = end_time
                if (i + 1) % 28000 is 0:
                    pn.save(sess, epoch)
            # compute val acc
            list_validation_accuracy = np.zeros(0, dtype=np.float)
            for i in range(5):
                test_index = i * 1000
                feed_dict={state: x_test[test_index:test_index + 1000], action: y_test[test_index:test_index + 1000], keep_prob: 1.0}
                validation_accuracy = sess.run(pn.accuracy, feed_dict)
                list_validation_accuracy = np.append(list_validation_accuracy, validation_accuracy)
            print("epoch: %d, average validation accuracy: %g" % (epoch, list_validation_accuracy.mean()))
