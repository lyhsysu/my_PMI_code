# _*_ coding: utf-8 _*_
"""
Time:       2022/12/08
Author:     Yuhao Li
Version:    V 0.1
File:       GAN_by_Keras.py.py
IDLE:       Pycharm
Copyright:  CC-BY-NA-SC
"""

from numpy import arange, expand_dims, array
from numpy.random import randn
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generator_model():
    # 搭建生成器
    model = Sequential()
    # 使用一个全连接层，输入为100维，输出为1000维
    model.add(Dense(1000, input_dim=100))
    # 激活函数使用tanh函数
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    # 搭建判别器
    model = Sequential()
    # 使用一个全连接层，输入为1000维，输出为1维
    model.add(Dense(1, input_dim=1000))
    # 激活函数使用sigmoid函数
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    # 将前面定义的生成器架构和判别器架构组拼接成一个大的神经网络
    model = Sequential()
    # 先添加生成器架构，再令d不可训练，即固定d
    # 因此在给定d的情况下训练生成器，即通过将生成的结果投入到判别器进行辨别而优化生成器
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def train():
    # 将定义好的模型架构赋值给特定的变量
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # 绘图
    g_loss_list = []
    d_loss_list = []
    g_data_list = []

    # 定义模型使用的优化算法及超参数
    # d_optim = SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)
    # g_optim = SGD(learning_rate=0.0009, momentum=0.9, nesterov=True)
    d_optim = Adam(learning_rate=0.0007)
    g_optim = Adam(learning_rate=0.0007)

    # 编译三个网络并设置损失函数和优化算法，其中损失函数都是用的是二元分类交叉熵函数。
    g.compile(loss='binary_crossentropy', optimizer="Adam")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    # 前一个架构在固定判别器的情况下训练了生成器，所以在训练判别器之前先要设定其为可训练。
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # 真实数据
    real_data = arange(1000)
    real_data[real_data < 100] = 1
    real_data[real_data != 1] = -1
    # shuffle(real_data)

    real_label = array([1])
    gen_label = array([0])

    # 对抗网络迭代次数
    for T in range(100):
        print("Epoch is", T)

        for K in range(100):
            # 随机生成100个标准高斯噪声
            noise = randn(100)
            # 使用生成器对随机噪声进行推断
            gen_data = g.predict(expand_dims(noise,axis=0), verbose=0).reshape(-1)
            gen_data[gen_data > 0] = 1
            gen_data[gen_data < 0] = -1

            # 判别器的损失
            d_gen_loss = d.train_on_batch(gen_data.reshape(1,-1), gen_label.reshape(1,-1))
            d_real_loss = d.train_on_batch(real_data.reshape(1,-1), real_label.reshape(1,-1))
            d_loss = d_gen_loss + d_real_loss

            # 随机生成100个标准高斯噪声
            noise = randn(100)
            # 固定判别器
            d.trainable = False

            # 计算生成器损失
            g_loss = d_on_g.train_on_batch(noise.reshape(1,-1), real_label.reshape(1,-1))

            # 令判别器可训练
            d.trainable = True

            # 统计生成数据
            gen_data_finally = g.predict(expand_dims(noise,axis=0), verbose=0)
            gen_data_finally[gen_data_finally > 0] = 1
            gen_data_finally[gen_data_finally < 0] = -1
            result = gen_data_finally[gen_data_finally == 1].size
            print("第%d次迭代，判别网络损失为：%f，生成网络损失为：%f" % (K, d_loss, g_loss))
            print("结果统计：生成数据中1的个数", result)
            if K == 99:
                d_loss_list.append(d_loss)
                g_loss_list.append(g_loss)
                g_data_list.append(result)

    return g_loss_list, d_loss_list, g_data_list


if __name__ == "__main__":
    g_loss_list, d_loss_list, g_data_list = train()

    plt.plot(range(len(g_loss_list)), g_loss_list, linewidth=1.5, c='red', label='loss of generator')
    plt.plot(range(len(g_loss_list)), d_loss_list, linewidth=1.5, c='blue', label='loss of discriminator')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='best', fontsize=16)
    plt.show()
    plt.clf()
    plt.plot(range(len(g_loss_list)), g_data_list, c='black', label='the number of 1 in the generated data.')
    plt.legend(loc='best', fontsize=13)
    plt.xlabel('epochs')
    plt.ylabel('the number of 1')
    plt.show()
