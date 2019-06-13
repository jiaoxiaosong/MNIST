# -*- coding: utf-8 -*-
# @Time    : 2019/6/13 上午9:50
# @Author  : baby松
# @FileName: vgg_17flower.py
# @Software: PyCharm

"""
vggnet 识别17种花
"""

import tensorflow as tf
from tflearn.datasets import oxflower17

# 读取数据
X, Y = oxflower17.load_data(dirname='17flowers', one_hot=True)
print(X.shape)
# 学习率
learn_rate = 0.1

# 每次迭代的训练样本数量
batch_size = 32

# 训练迭代次数（每个迭代次数必须训练完一次所有的数据）
train_epoch = 1000

# 样本数量
total_sample_number = X.shape[0]

# 模型构建
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, 17], name='y')


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


# 网络构建
def vgg_network(x, y):
    net1_kernel_size = 16
    net3_kernel_size = 32
    net5_kernel_size_1 = 64
    net5_kernel_size_2 = 64
    net7_kernel_size_1 = 128
    net7_kernel_size_2 = 128
    net9_kernel_size_1 = 128
    net9_kernel_size_2 = 128
    net11_util_size = 1024
    net12_util_size = 1024
    net13_util_size = 17

    # conv3-64
    with tf.variable_scope('net1'):
        net = tf.nn.conv2d(x, filter=get_variable('w', [3, 3, 3, net1_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net1_kernel_size]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

    # maxpool
    with tf.variable_scope('net2'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3_128
    with tf.variable_scope('net3'):
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net1_kernel_size, net3_kernel_size]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net3_kernel_size]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

    # maxpool
    with tf.variable_scope('net4'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3_256 conv3_256
    with tf.variable_scope('net5'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net3_kernel_size, net5_kernel_size_1]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net5_kernel_size_1]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net5_kernel_size_1, net5_kernel_size_2]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net5_kernel_size_2]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

    # maxpool
    with tf.variable_scope('net6'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3_512 conv3_512
    with tf.variable_scope('net7'):
        net = tf.nn.conv2d(net, filter=get_variable('w3', [3, 3, net5_kernel_size_2, net7_kernel_size_1]),
                           strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b3', [net7_kernel_size_1]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

        net = tf.nn.conv2d(net, filter=get_variable('w4', [3, 3, net7_kernel_size_1, net7_kernel_size_2]),
                           strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b4', [net7_kernel_size_2]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

    # maxpool
    with tf.variable_scope('net8'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3_512 conv3_512
    with tf.variable_scope('net9'):
        net = tf.nn.conv2d(net, filter=get_variable('w5', [3, 3, net7_kernel_size_2, net9_kernel_size_1]),
                           strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b5', [net9_kernel_size_1]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

        net = tf.nn.conv2d(net, filter=get_variable('w6', [3, 3, net9_kernel_size_1, net9_kernel_size_2]),
                           strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b6', [net9_kernel_size_2]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)

    # maxpool
    with tf.variable_scope('net10'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fc
    with tf.variable_scope('net11'):
        shape = net.get_shape()
        feature_number = shape[1] * shape[2] * shape[3]
        net = tf.reshape(net, shape=[-1, feature_number])
        net = tf.add(tf.matmul(net, get_variable('w7', [feature_number, net11_util_size])),
                     get_variable('b7', [net11_util_size]))

    # fc
    with tf.variable_scope('net12'):
        net = tf.add(tf.matmul(net, get_variable('w8', [net11_util_size, net12_util_size])),
                     get_variable('b8', [net12_util_size]))

    # fc
    with tf.variable_scope('net13'):
        net = tf.add(tf.matmul(net, get_variable('w9', [net12_util_size, net13_util_size])),
                     get_variable('b9', [net13_util_size]))

    # softmax
    with tf.variable_scope("net14"):
         act = tf.nn.softmax(net)
    return act


if __name__ == '__main__':
    act = vgg_network(x, y)
    # 构建模型的损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))
    # 梯度下降
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
    # 正确率
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    with tf.Session() as sess:
        # 初始化
        tf.global_variables_initializer().run()
        # 迭代训练
        for epoch in range(train_epoch):
            # 计算一次迭代batch执行的次数
            total_batch = int(total_sample_number / batch_size) - 5
            for step in range(total_batch):
                # 获取当前批次的数据
                train_x = X[step*batch_size:step*batch_size+batch_size]
                train_y = Y[step*batch_size:step*batch_size+batch_size]
                # 模型训练
                sess.run(train, feed_dict={x: train_x, y: train_y})

                # if step % 1 == 0:
                #     loss, accuracy = sess.run([cost, acc], feed_dict={x: train_x, y: train_y})
                #     print("训练集损失函数:{}, 训练集准确率:{}".format(loss, accuracy))
            if epoch % 10 == 0:
                test_x = X[step * batch_size]
                test_y = Y[step * batch_size]

                loss, accuracy = sess.run([cost, acc], feed_dict={x: test_x, y: test_y})
                print("测试集损失函数:{}, 测试集准确率:{}".format(loss, accuracy))

                loss, accuracy = sess.run([cost, acc], feed_dict={x: train_x, y: train_y})
                print("训练集损失函数:{}, 训练集准确率:{}".format(loss, accuracy))