# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 下午4:43
# @Author  : baby松
# @FileName: softmax_mnist.py
# @Software: PyCharm
import ssl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
ssl._create_default_https_context = ssl._create_unverified_context
mnist = input_data.read_data_sets('.//MNIST/MNIST_data', one_hot=True)
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


batch_xs, batch_ys = mnist.train.next_batch(100)
train_step = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print(train_step)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
a = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(a)
