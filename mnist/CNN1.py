'''
Model_CNN1 doesn't divide training set into training set & validation set.
'''

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import xavier_initializer

# Setting dataset
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

x_train = train.iloc[:,1:].values.astype(np.float32)
y_train = train.iloc[:,0].values.astype(np.int32)
x_test = test.iloc[:,:].values.astype(np.float32)

# Hyper Parameter
learning_rate = 0.001
epochs = 15
batch_size = 100

class Model_CNN1:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        ''' CNN model. Input -> [ [CONV -> RELU] * 1 -> POOL ] * 2 -> FC'''

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        x_img = tf.reshape(self.x, [-1, 28, 28, 1])
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
        self.yonehot = tf.one_hot(self.y, depth=10)
        # tf_y = tf.placeholder(dtype=tf.int32, shape=[None, nb_classes], name='tf_y')

        # Layer 1. filter = 3 * 3, strides = 2 * 2, depth=32
        W1 = tf.get_variable(name='W1', shape=[3, 3, 1, 32], initializer=xavier_initializer())
        # conv -> [-1, 28, 28, 32]
        # pool -> [-1, 14, 14, 32]
        L1 = tf.nn.conv2d(x_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

        # Layer2. filter = 3 * 3, strides = 2 * 2, depth=64
        W2 = tf.get_variable(name='W2', shape=[3, 3, 32, 64], initializer=xavier_initializer())
        # conv -> [-1, 14, 14, 64]
        # pool -> [-1, 7, 7, 64]
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
        L2 = tf.reshape(L2, [-1, 7*7*64])

        # Final FC 7*7*64 input -> 10 output
        W3 = tf.get_variable(name='W3', shape=[7*7*64, 10], initializer=xavier_initializer())
        b3 = tf.Variable(tf.random_normal([10]))
        self.logits = tf.matmul(L2, W3) + b3

        # Define cost & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.yonehot))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.yonehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def prediction(self, x_test, keep_prob=1.0):
        return tf.argmax(self.sess.run(self.logits, feed_dict={self.x: x_test,
                self.keep_prob: keep_prob}), 1)

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test,
                self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
                self.x: x_data, self.yonehot: y_data, self.keep_prob: keep_prob})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './trained_model/model_CNN1')

# define a function to creat batch generator
def creat_batch_generator(x, y, sess, batch_size=batch_size):
    x_copy = np.copy(x)
    y_copy = np.copy(y)
    with sess.as_default():
        y_onehot = tf.one_hot(y_copy, depth=10).eval()
        for i in range(0, x.shape[0], batch_size):
            yield (x_copy[i:i + batch_size, :], y_onehot[i:i + batch_size, :])

sess = tf.Session()
cnn = Model_CNN1(sess, 'cnn')

sess.run(tf.global_variables_initializer())

print('Learning started(%2d epochs). It takes sometime.\n' %(epochs))

for epoch in range(epochs):

    batch_generator = creat_batch_generator(x_train, y_train, sess)

    avg_cost = 0
    for batch_x, batch_y in batch_generator:
        cost_, _ = cnn.train(batch_x, batch_y)
        avg_cost += cost_ / int(x_train.shape[0] / batch_size)

    accuracy = cnn.get_accuracy(x_train, y_train)
    print('epoch %2d   training cost: %.6f  accruacy: %.2f'\
            %(epoch + 1, avg_cost, 100*accuracy ))

print('\nLearning finished!')

cnn.save()

with sess.as_default():
    prediction = cnn.prediction(x_test).eval()

submission = pd.DataFrame({
    'ImageId' : range(1, x_test.shape[0] + 1),
    'Label' : prediction
})
submission.to_csv('./submission/submission_cnn1.csv', index=None)


'''
the result is as follow:

epoch  1   training cost: 1.489410  accruacy: 96.36
epoch  2   training cost: 0.183580  accruacy: 98.01
epoch  3   training cost: 0.124849  accruacy: 98.49
epoch  4   training cost: 0.100465  accruacy: 98.74
epoch  5   training cost: 0.088716  accruacy: 98.97
epoch  6   training cost: 0.079759  accruacy: 99.12
epoch  7   training cost: 0.071472  accruacy: 99.15
epoch  8   training cost: 0.069878  accruacy: 99.09
epoch  9   training cost: 0.063442  accruacy: 99.13
epoch 10   training cost: 0.062499  accruacy: 99.32
epoch 11   training cost: 0.061118  accruacy: 99.33
epoch 12   training cost: 0.055946  accruacy: 99.36
epoch 13   training cost: 0.054146  accruacy: 99.42
epoch 14   training cost: 0.054928  accruacy: 99.44
epoch 15   training cost: 0.052440  accruacy: 99.39

Before this model, I built a model with training data and validation data.
Even though the accuracy with training data got better, prediction of kaggle test data
changed little. 
'''
