# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 下午3:26
# @Author  : baby松
# @FileName: cnn_mnist_lenet.py
# @Software: PyCharm
"""
手写数字识别的CNN网络 LeNet
"""

import ssl
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

ssl._create_default_https_context = ssl._create_unverified_context
mnist = input_data.read_data_sets('/Users/xiaosongzi/Desktop/learn_project/MNIST/MNIST/MNIST_data/', one_hot=True)

train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
train_sample_number = mnist.train.num_examples

# 相关的参数、超参数的设置
# 学习率
learn_rate = 1e-2
# 训练的迭代次数
train_epoch = 10000
# 每次迭代的训练样本的数量
batch_size = 64
display_step = 40

# 输入的样本大小信息
input_dim = train_img.shape[1]
# 输出的维度大小信息
n_classes = train_label.shape[1]

# 模型构建
# 1、设置数据输入的占位符
x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')


def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    """
    返回一个对应的变量
    :param name:
    :param shape:
    :param dtype:
    :param initializer:
    :return:
    """
    return tf.get_variable(name, shape, dtype, initializer)


# 2、构建网络
def le_net(x, y):
    # 1、输入层
    with tf.variable_scope("input"):
        # 将输入的x的格式转换为规定的格式
        # [None, input_dim] -> [None, height, weights, champles]
        net = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 2、卷积层
    with tf.variable_scope('conv1'):
        # padding-same:填充；valid:丢弃多余的特征；
        net = tf.nn.conv2d(net, filter=get_variable('w', [5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')
        # net = tf.nn.bias_add(net, get_variable('b', [20]))
        # 激励 Relu
        net = tf.nn.relu(net)

    # 3、池化
    with tf.variable_scope('pool3'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #===========================================================================================

    # 4、卷积层
    with tf.variable_scope('sonv4'):
        net = tf.nn.conv2d(net, filter=get_variable('w', [5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')
        # net = tf.nn.bias_add(net, get_variable('b', [50]))
        # 激励Relu
        net = tf.nn.relu(net)

    # 5、池化
    with tf.variable_scope('pool5'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 6、全连接
    with tf.variable_scope('fc6'):
        net = tf.reshape(net, shape=[-1, 7 * 7 * 50])
        net = tf.matmul(net, get_variable('w', [7 * 7 * 50, 500]))
        # net = tf.add(tf.matmul(net, get_variable('w', [7 * 7 * 50, 500])), get_variable('b', [500]))
        net = tf.nn.relu(net)

    # 7、全链接
    with tf.variable_scope('fc7'):
        net = tf.matmul(net, get_variable('w', [500, n_classes]))
        # net = tf.add(tf.matmul(net, get_variable('w', [500, n_classes])), get_variable('b', [n_classes]))
        act = tf.nn.softmax(net)
    return act


if __name__ == '__main__':
    act = le_net(x, y)
    # 构建模型的损失函数
    # softmax_cross_entropy_with_logits 计算softmax每个样本的交叉熵，logits指定预测值，labels指定实际值
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))
    # 使用梯度下降求解
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
    # 正确率
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    # 初始化
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 进行数据初始化
        sess.run(init)
        # 模型保存、持久化
        saver = tf.train.Saver()
        epoch = 0
        while True:
            avg_cost = 0
            # 计算出总的批次
            total_batch = int(mnist.train.num_examples / batch_size)
            total_batch = 100
            for i in range(total_batch):
                # 获取x和y
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feeds = {x: batch_xs, y: batch_ys}
                # 模型训练
                sess.run(train, feed_dict=feeds)
                # 获取损失函数值
                avg_cost += sess.run(cost, feed_dict=feeds)

            # 重新计算平均损失
            avg_cost = avg_cost / total_batch

            if (epoch + 1) % display_step == 0:
                print("批次：%03d 损失函数值: %.9f" % (epoch, avg_cost))
                feeds = {
                    x: batch_xs,
                    y: batch_ys,
                }
                train_acc = sess.run(acc, feed_dict=feeds)
                print("训练集准确率:%.3f" % train_acc)

                feeds = {
                    x: test_img,
                    y: test_label,

                }
                test_acc = sess.run(acc, feed_dict=feeds)
                print("测试准确率:%.3f" % test_acc)

                if train_acc > 0.9 and test_acc > 0.9:
                    saver.save(sess, '/Users/xiaosongzi/Desktop/learn_project/MNIST/model/')
                    break
            epoch += 1