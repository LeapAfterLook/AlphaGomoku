import tensorflow as tf
import numpy as np
from ..ParseGomoku.renjunet import Record
import os

def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d_same(input, filter):
    return tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')


cur_dir = os.path.abspath(os.path.dirname(__file__))

# load data
x_board_images = Record.load_input_records()
y_board_labels = abs(Record.load_output_records())

x_train, x_test = x_board_images[:1400000], x_board_images[1400000:]
y_train, y_test = y_board_labels[:1400000], y_board_labels[1400000:]

# build model origin
sess = tf.Session()
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.int8, [None, 15, 15])
x_board = tf.cast(tf.reshape(x, [-1, 15, 15, 1]), tf.float32)
y_answer = tf.placeholder(tf.int8, [None, 2, 15])
y_row_answer, y_col_answer = tf.split(tf.reshape(y_answer, [-1, 30]), 2, 1)

# 1st conv layer
W_conv1 = weight_variable("W_conv1", [5, 5, 1, 1024])
b_conv1 = bias_variable("b_conv1", [1024])

h_conv1 = tf.nn.relu(conv2d_same(x_board, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

# 2nd conv layer
W_conv2 = weight_variable("W_conv2", [3, 3, 1024, 512])
b_conv2 = bias_variable("b_conv2", [512])

h_conv2 = tf.nn.relu(conv2d_same(h_conv1_drop, W_conv2) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)

# 3rd conv layer
W_conv3 = weight_variable("W_conv3", [3, 3, 512, 256])
b_conv3 = bias_variable("b_conv3", [256])

h_conv3 = tf.nn.relu(conv2d_same(h_conv2_drop, W_conv3) + b_conv3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

# 4th conv layer
W_conv4 = weight_variable("W_conv4", [3, 3, 256, 256])
b_conv4 = bias_variable("b_conv4", [256])

h_conv4 = tf.nn.relu(conv2d_same(h_conv3_drop, W_conv4) + b_conv4)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)

h_conv4_flat = tf.reshape(h_conv4_drop, [-1, 15 * 15 * 256])

# 1st fc layer
W_fc1 = weight_variable("W_fc1", [15 * 15 * 256, 1024])
b_fc1 = bias_variable("b_fc1", [1024])

h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd fc layer
W_fc2 = weight_variable("W_fc2", [1024, 256])
b_fc2 = bias_variable("b_fc2", [256])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 3rd fc layer
W_fc3 = weight_variable("W_fc3", [256, 30])
b_fc3 = bias_variable("b_fc3", [30])

y = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

y_row_pred, y_col_pred = tf.split(y, 2, 1)

saver = tf.train.Saver()

row_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_row_answer, logits=y_row_pred)
col_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_col_answer, logits=y_col_pred)
cross_entropy = tf.reduce_mean(tf.concat([row_cross_entropy, col_cross_entropy], 0))

# train & evaluate the model
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
row_equal = tf.equal(tf.argmax(y_row_pred, 1), tf.argmax(y_row_answer, 1))
col_equal = tf.equal(tf.argmax(y_col_pred, 1), tf.argmax(y_col_answer, 1))
correct_prediction = row_equal & col_equal
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#    saver.restore(sess, os.path.join(cur_dir, 'tmp_no_pooling/6_28000_model.ckpt'))
#    print("Model restored.")
    for epoch in range(1, 30):
        if epoch == 31:
            for i in range(8000, 28000):
                batch_start = i * 50
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict={x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 0.5})
                if (i + 1) % 1000 == 0 and i > 10:
                    save_path = saver.save(sess, cur_dir + '/tmp/' + str(epoch) + '_' + str(i + 1) + '_model.ckpt')
                    print("Model saved in file: %s" % save_path)
        else:
            train = list(zip(x_train, y_train))
            np.random.shuffle(train)
            x_train, y_train = zip(*train)
            for i in range(28000):
                batch_start = i * 50
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 1.0})
                    loss = cross_entropy.eval(feed_dict={x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 1.0})
                    print('step %d, training accuracy %g, loss %g' % (i, train_accuracy, loss))
                sess.run(train_step, feed_dict={x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 0.5})
                if (i + 1) % 28000 == 0:
                   save_path = saver.save(sess, cur_dir + '/tmp_4conv3fc__no_pooling/' + str(epoch) + '_' + str(i + 1) + '_model.ckpt')
                   print("Model saved in file: %s" % save_path)
        list_test_accuracy = np.zeros(0, dtype=np.float)
        for i in range(5):
            test_index = i * 1000
            test_accuracy = accuracy.eval(feed_dict={x: x_test[test_index:test_index + 1000], y_answer: y_test[test_index:test_index + 1000], keep_prob: 1.0})
            list_test_accuracy = np.append(list_test_accuracy, test_accuracy)
            print('epoch: %d, test accuracy: %g' % (epoch, test_accuracy))
        print("average test accuracy: %g" % list_test_accuracy.mean())

