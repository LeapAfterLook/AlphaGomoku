import tensorflow as tf
import numpy as np
import os


class PolicyNetwork:
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    def __init__(self, dic_placeholder, name="policy_net", learning_rate=1e-4):
        self.name = name
        self.restored = False
        with tf.variable_scope(name):
            self.state = dic_placeholder['state']  # assumes shape of [batch_size, 15, 15, 2], dtype of tf.int8 and name of "state"
            self.state_cast = tf.cast(self.state, tf.float32)
            self.action = dic_placeholder['action'] # assumes shape of [batch_size, 225], dtype of tf.int8 and name of "action"
            self.action_cast = tf.cast(self.action, tf.float32) 
            self.keep_prob = dic_placeholder['keep_prob']
            
            # This is CNN + FC estimator
            self.conv1 = tf.contrib.layers.conv2d(inputs=self.state_cast, num_outputs=256, kernel_size=5, scope="conv1", biases_initializer=tf.constant_initializer(0.1))
            self.conv1_drop = tf.nn.dropout(self.conv1, keep_prob=self.keep_prob)
            self.conv2 = tf.contrib.layers.conv2d(inputs=self.conv1_drop, num_outputs=256, kernel_size=3, scope="conv2", biases_initializer=tf.constant_initializer(0.1))
            self.conv2_drop = tf.nn.dropout(self.conv2, keep_prob=self.keep_prob)
            self.conv3 = tf.contrib.layers.conv2d(inputs=self.conv2_drop, num_outputs=256, kernel_size=3, scope="conv3", biases_initializer=tf.constant_initializer(0.1))
            self.conv3_drop = tf.nn.dropout(self.conv3, keep_prob=self.keep_prob)
            self.conv4 = tf.contrib.layers.conv2d(inputs=self.conv3_drop, num_outputs=256, kernel_size=3, scope="conv4", biases_initializer=tf.constant_initializer(0.1))
            self.conv4_drop = tf.nn.dropout(self.conv4, keep_prob=self.keep_prob)
            self.conv4_flat = tf.contrib.layers.flatten(self.conv4_drop)
            self.fc5 = tf.contrib.layers.fully_connected(inputs=self.conv4_flat, num_outputs=512, scope="fc5", biases_initializer=tf.constant_initializer(0.1))
            self.fc5_drop = tf.nn.dropout(self.fc5, keep_prob=self.keep_prob)
            self.fc6 = tf.contrib.layers.fully_connected(inputs=self.fc5_drop, num_outputs=225, activation_fn=None, scope="fc6", biases_initializer=tf.constant_initializer(0.1))
            self.policy = tf.nn.softmax(logits=self.fc6)
            
            # stage the model structure to the saver
            self.saver = tf.train.Saver()
            
            # loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.action_cast, logits=self.fc6))
            
            # gradient & accuracy
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            self.correct_prediction = tf.equal(tf.argmax(self.policy, 1), tf.argmax(self.action, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def save(self, sess, epoch):
        save_dir = os.path.join(self.model_dir, self.name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.save_path = self.saver.save(sess, os.path.join(save_dir, "epoch_%s_model.ckpt" % str(epoch)))
        print("Model saved in %s." % self.save_path)

    def restore(self, sess, epoch):
        save_dir = os.path.join(self.model_dir, self.name)
        self.saver.restore(sess, os.path.join(save_dir, "epoch_%s_model.ckpt" % str(epoch)))
        self.restored = True

    def inference(self, sess, board):
        if not self.restored:
            print("Please restore the model")
            exit(1)
        feed_dict = {self.state: board.reshape(-1, 15, 15, 2), self.keep_prob: 1.0}
        prediction = sess.run(self.policy, feed_dict)
        return prediction
