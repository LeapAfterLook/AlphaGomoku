import tensorflow as tf
from ParseGomoku.renjunet import Record

def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d_same(input, filter):
    return tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(value):
    return tf.nn.max_pool(value=value, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')


# load data
x_board_images = Record.load_input_records()
y_board_labels = abs(Record.load_output_records())

x_train, x_test = x_board_images[:1000000], x_board_images[1000000:]
y_train, y_test = y_board_labels[:1000000], y_board_labels[1000000:]

# build model origin
sess = tf.Session()
saver = tf.train.Saver()
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.int8, [None, 15, 15])
x_board = tf.reshape(x, [-1, 15, 15, 1])
y_answer = tf.reshape(tf.placeholder(tf.int8, [None, 2, 15]), [-1, 30])
y_row_answer, y_col_answer = tf.split(y_answer, 2, 1)

# 1st conv layer
W_conv1 = weight_variable("W_conv1", [5, 5, 1, 1024])
b_conv1 = bias_variable("b_conv1", [1024])

h_conv1 = tf.nn.relu(conv2d_same(x_board, W_conv1) + b_conv1)
h_pool1 = tf.max_pool_3x3(h_conv1)

# 2nd conv layer
W_conv2 = weight_variable("W_conv2", [3, 3, 1, 256])
b_conv2 = bias_variable("b_conv2", [256])

h_conv2 = tf.nn.relu(conv2d_same(h_pool1, W_conv2) + b_conv2)

# 3rd conv layer
W_conv3 = weight_variable("W_conv3", [3, 3, 1, 128])
b_conv3 = bias_variable("b_conv3", [128])

h_conv3 = tf.nn.relu(conv2d_same(h_conv2, W_conv3) + b_conv3)

# 1st fc layer
W_fc1 = weight_variable([5 * 5 * 128, 1024])
b_fc1 = bias_variable([1024])

h_conv3_flat = tf.reshape(h_conv3, [-1, 5 * 5 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd fc layer
W_fc2 = weight_variable([1024, 30])
b_fc2 = bias_variable([30])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_row_pred, y_col_pred = tf.split(y, 2, 1)

# train & evaluate the model
row_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_row_answer, logits=y_row_pred)
col_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_col_answer, logits=y_col_pred)
cross_entropy = tf.reduce_mean(tf.concat([row_cross_entropy, col_cross_entropy], 0))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
row_equal = tf.equal(tf.argmax(y_row_pred, 1), tf.argmax(y_row_answer, 1))
col_equal = tf.equal(tf.argmax(y_col_pred, 1), tf.argmax(y_col_answer, 1))
correct_prediction = row_equal & col_equal
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_start = i * 50
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        sess.run(train_step, feed_dict={
                x: x_train[batch_start:batch_start + 50], y_answer: y_train[batch_start:batch_start + 50], keep_prob: 0.5})
        if i + 1 % 5000 == 0 and i > 10:
            save_path = saver.save(sess, '/tmp/' + str(i) + '_model.ckpt')
            print("Model saved in file: %s" % save_path)

    print('test accuracy %g' % accuracy.eval(feed_dict={
                x: x_test[:1000], y_answer: y_test[:1000], keep_prob: 1.0}))
