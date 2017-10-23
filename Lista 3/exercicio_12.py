import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar_array():
    path_file = os.getcwd() + "\\Lista 3\\cifar-10-batches-py\\"
    cifar_array = []
    for i in range(5):
        train_batch = unpickle(path_file + "data_batch_" + str(i + 1))
        cifar_array.append(train_batch)

    test_batch = unpickle(path_file + "test_batch")
    cifar_array.append(test_batch)
    return cifar_array

#train_input, train_label = cifar_array(os.getcwd() + "\\Lista 3\\cifar-100-python\\train")
#test_input, test_label = cifar_array(os.getcwd() + "\\Lista 3\\cifar-100-python\\test")

cifar_array = cifar_array()

array_train = [cifar_array[0], cifar_array[1], cifar_array[2], cifar_array[3], cifar_array[4]]
array_test = cifar_array[5]

TAXA_APRENDIZADO = 0.01
QUANTIDADE_EPOCAS = 50
#TAMANHO_LOTE = 100

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 3072])
x_reshape = tf.reshape(x, [-1, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([3072, 10]))
b = tf.Variable(tf.zeros([10]))

# (x * W) + b
y = tf.matmul(x, W) + b

custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
otimizador = tf.train.AdamOptimizer(learning_rate=TAXA_APRENDIZADO).minimize(custo)

sess.run(tf.global_variables_initializer())

for i in range(QUANTIDADE_EPOCAS):
    lista_custos = []
    total_batch = len(array_train)

    for j in range(total_batch):
        #batch = array_train.next_batch(TAMANHO_LOTE)
        batch = array_train[j]
        array_labels = []
        for k in range(len(batch[b'labels'])):
            label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label[batch[b'labels'][k]] = 1
            array_labels.append(label)

        custo_result, _ = sess.run([custo, otimizador], feed_dict={x: batch[b'data'], y_: array_labels})
        lista_custos.append(custo_result)

    print('Epoca:', '%04d' % (i + 1), 'perda =', '{:.9f}'.format(np.average(lista_custos)))

predicao = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predicao, tf.float32))

array_labels = []

for i in range(len(array_test[b'labels'])):
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label[array_test[b'labels'][k]] = 1
    array_labels.append(label)

print(accuracy.eval(feed_dict={x: array_test[b'data'], y_: array_labels}))
