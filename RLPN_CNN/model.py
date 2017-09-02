import tensorflow as tf
import numpy as np
import os
import itertools
import time

from ..util.UserInterface.board import plot_board, plot_action_prediction
from ..util.GameManager.GomokuManager import GomokuManager


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1))


class PolicyNetwork:
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    def __init__(self, dic_placeholder, name="policy_net", learning_rate=1e-4):
        self.name = name
        self.restored = False
        with tf.variable_scope(name) as scope:
            self.state = dic_placeholder['state']  # assumes shape of [batch_size, 15, 15, 2], dtype of tf.int8 and name of "state"
            self.state_cast = tf.cast(self.state, tf.float32)
            self.action = dic_placeholder['action'] # assumes shape of [batch_size, 225], dtype of tf.int8 and name of "action"
            self.action_cast = tf.cast(self.action, tf.float32) 
            self.target = dic_placeholder['target'] # assumes shape of [batch_size] dtype of tf.float32 and name of "target"
            self.target_expand = tf.expand_dims(self.target, 1)
            self.keep_prob = dic_placeholder['keep_prob']
        
            with tf.Graph().as_default() as self.graph:
                # This is CNN + FC estimator
                with tf.variable_scope("conv1"):
                    self.W_conv1 = weight_variable("weights", [5, 5, 2, 256])
                    self.b_conv1 = bias_variable("biases", [256])
                    self.conv1 = tf.nn.relu(tf.nn.conv2d(self.state_cast, self.W_conv1, [1, 1, 1, 1], 'SAME') + self.b_conv1)
                self.conv1_drop = tf.nn.dropout(self.conv1, keep_prob=self.keep_prob)
                with tf.variable_scope("conv2"):
                    self.W_conv2 = weight_variable("weights", [3, 3, 256, 256])
                    self.b_conv2 = bias_variable("biases", [256])
                    self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1_drop, self.W_conv2, [1, 1, 1, 1], 'SAME') + self.b_conv2)
                self.conv2_drop = tf.nn.dropout(self.conv2, keep_prob=self.keep_prob)
                with tf.variable_scope("conv3"):
                    self.W_conv3 = weight_variable("weights", [3, 3, 256, 256])
                    self.b_conv3 = bias_variable("biases", [256])
                    self.conv3 = tf.nn.relu(tf.nn.conv2d(self.conv2_drop, self.W_conv3, [1, 1, 1, 1], 'SAME') + self.b_conv3)
                self.conv3_drop = tf.nn.dropout(self.conv3, keep_prob=self.keep_prob)
                with tf.variable_scope("conv4"):
                    self.W_conv4 = weight_variable("weights", [3, 3, 256, 256])
                    self.b_conv4 = bias_variable("biases", [256])
                    self.conv4 = tf.nn.relu(tf.nn.conv2d(self.conv3_drop, self.W_conv4, [1, 1, 1, 1], 'SAME') + self.b_conv4)
                self.conv4_drop = tf.nn.dropout(self.conv4, keep_prob=self.keep_prob)
                self.conv4_flat = tf.contrib.layers.flatten(self.conv4_drop)
                with tf.variable_scope("fc5"):
                    self.W_fc5 = weight_variable("weights", [15 * 15 * 256, 512])
                    self.b_fc5 = bias_variable("biases", [512])
                    self.fc5 = tf.nn.relu(tf.matmul(self.conv4_flat, self.W_fc5) + self.b_fc5)
                self.fc5_drop = tf.nn.dropout(self.fc5, keep_prob=self.keep_prob)
                with tf.variable_scope("fc6"):
                    self.W_fc6 = weight_variable("weights", [512, 225])
                    self.b_fc6 = bias_variable("biases", [225])
                    self.fc6 = tf.matmul(self.fc5_drop, self.W_fc6) + self.b_fc6
                self.policy = tf.nn.softmax(logits=self.fc6)            
                # stage the model structure to the saver
                self.saver = tf.train.Saver()
                # loss
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.action_cast * self.target_expand, logits=self.fc6))
                # gradient & accuracy
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    
    def save(self, sess, episode):
        save_dir = os.path.join(self.model_dir, self.name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.save_path = self.saver.save(sess, os.path.join(save_dir, "episode_%s_model.ckpt" % str(episode)))
        print("Model saved in %s." % self.save_path)

    def restore(self, sess, episode):
        save_dir = os.path.join(self.model_dir, self.name)
        self.saver.restore(sess, os.path.join(save_dir, "episode_%s_model.ckpt" % str(episode)))
        self.restored = True
    
    def restore_sl_pn(self, sess, epoch):
        save_dir = os.path.join(os.path.dirname(self.model_dir), "SLPN_CNN", self.name)
        self.saver.restore(sess, os.path.join(save_dir, "epoch_%s_model.ckpt" % str(epoch)))    
        self.restored = True
    
    def inference(self, sess, board):
        if not self.restored:
            print("Please restore the model")
            exit(1)
        feed_dict = {self.state: board.reshape(-1, 15, 15, 2), self.keep_prob: 1.0}
        prediction = sess.run(self.policy, feed_dict)
        return prediction
