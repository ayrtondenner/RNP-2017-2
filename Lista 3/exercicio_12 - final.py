import datetime

inicio = datetime.datetime.now()

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from more_itertools import chunked

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

def calcula_acuracia_treino(array_train_images, array_train_labels):
    chunked_train_images = list(chunked(array_train_images, BATCH))
    chunked_train_labels = list(chunked(array_train_labels, BATCH))

    accuracy_list = []

    for i in range(len(chunked_train_images)):
        chunk_images = chunked_train_images[i]
        chunk_labels = chunked_train_labels[i]

        calculated_accuracy = acuracia.eval(feed_dict={x: chunk_images, y_real: chunk_labels,
        #keep_prob: KEEP_PROB_TEST
        })
        accuracy_list.append(calculated_accuracy)
        #print(calculated_accuracy)

    return accuracy_list

def calcula_acuracia_teste(array_test_images, array_test_labels):
    chunked_test_images = list(chunked(array_test_images, BATCH))
    chunked_test_labels = list(chunked(array_test_labels, BATCH))

    accuracy_list = []

    for i in range(len(chunked_test_images)):
        chunk_images = chunked_test_images[i]
        chunk_labels = chunked_test_labels[i]

        calculated_accuracy = acuracia.eval(feed_dict={x: chunk_images, y_real: chunk_labels
        #, keep_prob: KEEP_PROB_TEST
        })
        accuracy_list.append(calculated_accuracy)
        #print(calculated_accuracy)
        
    return accuracy_list

cifar_array = cifar_array()

array_train_images = []
array_train_labels = []
array_test_images = []
array_test_labels = []

for cifar_batch in cifar_array:

    for image in cifar_batch[b'data']:
        if 'training' in str(cifar_batch[b'batch_label']):
            array_train_images.append(image)
        elif 'testing' in str(cifar_batch[b'batch_label']):
            array_test_images.append(image)

    for label in cifar_batch[b'labels']:
        fixed_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fixed_label[label] = 1

        if 'training' in str(cifar_batch[b'batch_label']):
            array_train_labels.append(fixed_label)
        elif 'testing' in str(cifar_batch[b'batch_label']):
            array_test_labels.append(fixed_label)
'''
figure = plt.figure()

for i in range(50):
    subplot = figure.add_subplot(10, 5, i + 1)
    imagem = array_train_images[i]
    array_imagem = np.array([imagem])
    array_imagem = array_imagem.reshape(len(array_imagem), 3, 32, 32)
    array_imagem = array_imagem.transpose(0, 2, 3, 1)
    plt.imshow(array_imagem[0])

plt.show()
'''

TAXA_APRENDIZADO = 0.001
EPOCAS = 20
BATCH = 320
KEEP_PROB_TRAIN = 0.25
KEEP_PROB_TEST = 1.0

x = tf.placeholder(tf.float32, shape=[None, 3072]) # Tamanho total de uma imagem 32 x 32 x 3
x_reshape = tf.reshape(x, [-1, 3, 32, 32])
x_reshape = tf.transpose(x_reshape, [0, 2, 3, 1])
y_real = tf.placeholder(tf.float32, shape=[None, 10]) # Total de 10 classes no CIFAR-10

# LAYER 1 - Convolutional, ReLU, Convolutional, ReLU and Pooling

W1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 48], stddev=0.1)) # 32 features de tamanho 5 x 5
b1_1 = tf.Variable(tf.constant(0.1, shape=[48])) # bias para as 32 features

L1 = tf.nn.conv2d(x_reshape, W1_1, strides=[1, 1, 1, 1], padding='SAME') + b1_1
L1 = tf.nn.relu(L1)

W1_2 = tf.Variable(tf.truncated_normal([3, 3, 48, 48], stddev=0.1)) # 32 features de tamanho 5 x 5
b1_2 = tf.Variable(tf.constant(0.1, shape=[48])) # bias para as 32 features

L1 = tf.nn.conv2d(L1, W1_2, strides=[1, 1, 1, 1], padding='SAME') + b1_2
L1 = tf.nn.relu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#L1 = tf.nn.dropout(L1, keep_prob=KEEP_PROB_TRAIN)

# LAYER 2 - Convolutional, ReLU, Convolutional, ReLU and Pooling

W2_1 = tf.Variable(tf.truncated_normal([3, 3, 48, 128], stddev=0.1)) # 64 features de tamanho 5 x 5
b2_1 = tf.Variable(tf.constant(0.1, shape=[128])) # bias para as 64 features

L2 = tf.nn.conv2d(L1, W2_1, strides=[1, 1, 1, 1], padding='SAME') + b2_1
L2 = tf.nn.relu(L2)

W2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)) # 64 features de tamanho 5 x 5
b2_2 = tf.Variable(tf.constant(0.1, shape=[128])) # bias para as 64 features

L2 = tf.nn.conv2d(L2, W2_2, strides=[1, 1, 1, 1], padding='SAME') + b2_2
L2 = tf.nn.relu(L2)

L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#L2 = tf.nn.dropout(pool2, keep_prob=KEEP_PROB_TRAIN)

# LAYER 3 - Convolutional, ReLU, Convolutional, ReLU and Pooling

W3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)) # 64 features de tamanho 5 x 5
b3_1 = tf.Variable(tf.constant(0.1, shape=[128])) # bias para as 64 features

L3 = tf.nn.conv2d(L2, W3_1, strides=[1, 1, 1, 1], padding='SAME') + b3_1
L3 = tf.nn.relu(L3)

W3_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)) # 64 features de tamanho 5 x 5
b3_2 = tf.Variable(tf.constant(0.1, shape=[128])) # bias para as 64 features

L3 = tf.nn.conv2d(L3, W3_2, strides=[1, 1, 1, 1], padding='SAME') + b3_2
L3 = tf.nn.relu(L3)

L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#L3 = tf.nn.dropout(L3, keep_prob=KEEP_PROB_TRAIN)

# LAYER 4 - Fully Connected Layer

Wfc1 = tf.Variable(tf.truncated_normal([4 * 4 * 128, 1024], stddev=0.1)) # 64 features foram feitas, e o tamanho final do último layer fora 8 x 8
bfc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # 32 x 32 x 3

L4 = tf.reshape(L3, [-1, 4 * 4 * 128])
L4 = tf.matmul(L4, Wfc1) + bfc1
L4 = tf.nn.relu(L4)

#L4 = tf.nn.dropout(L4, keep_prob=KEEP_PROB_TRAIN)

# LAYER 5 - Readout

Wrl = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) # conectando os 3072 valores para as 10 classes do CIFAR-10
brl = tf.Variable(tf.constant(0.1, shape=[10])) # 10 classes

y_calculado = tf.matmul(L4, Wrl) + brl

custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=y_calculado))
otimizador = tf.train.AdamOptimizer(learning_rate=TAXA_APRENDIZADO).minimize(custo)
predicao = tf.equal(tf.arg_max(y_real, 1), tf.arg_max(y_calculado, 1))
acuracia = tf.reduce_mean(tf.cast(predicao, tf.float32))

with tf.Session() as session:

    print("Início do treinamento")

    session.run(tf.global_variables_initializer())

    # TREINO

    i = 0
    stop_training = False # Variável usada para encerrar o treino durante o debug

    lista_epocas = []
    lista_custos = []

    while i < EPOCAS and stop_training == False:

        inicio_epoca = datetime.datetime.now()

        cost_list = []

        chunked_train_images = list(chunked(array_train_images, BATCH))
        chunked_train_labels = list(chunked(array_train_labels, BATCH))

        for j in range(len(chunked_train_images)): # o tamanho do array de imagens e labels é o mesmo
            chunk_images = chunked_train_images[j]
            chunk_labels = chunked_train_labels[j]

            cost_result, _ = session.run([custo, otimizador], feed_dict={x: chunk_images, y_real: chunk_labels
            #, keep_prob: KEEP_PROB_TRAIN
            })
            cost_list.append(cost_result)

        epoca_atual = i + 1
        lista_epocas.append(epoca_atual)

        custo_atual = np.average(cost_list)
        lista_custos.append(custo_atual)

        mensagem_epoca = 'Época: ' + str(epoca_atual) + ', custo: ' + str(custo_atual)

        if (i % 10 == 0):
            acuracia_treino = np.average(calcula_acuracia_treino(array_train_images, array_train_labels))
            acuracia_teste = np.average(calcula_acuracia_teste(array_test_images, array_test_labels))
            mensagem_epoca = mensagem_epoca + ', acurácia do treino: ' + str(acuracia_treino) + ', acurácia do teste: ' + str(acuracia_teste)

        tempo_epoca = datetime.datetime.now() - inicio_epoca
        mensagem_epoca = mensagem_epoca + '\nTempo de treino: ' + str(tempo_epoca) + '\n'

        print(mensagem_epoca)

        i += 1

    # RESULTADO DAS ACURÁCIAS

    acuracia_treino = np.average(calcula_acuracia_treino(array_train_images, array_train_labels))
    acuracia_teste = np.average(calcula_acuracia_teste(array_test_images, array_test_labels))

    mensagem_resultado_final = 'Taxa de aprendizado: ' + str(TAXA_APRENDIZADO) + ', batch: ' + str(BATCH) + ', custo final: ' + str(lista_custos[-1]) + ', acurácia do treino: ' + str(acuracia_treino) + ', acurácia do teste: ' + str(acuracia_teste)
    print(mensagem_resultado_final)
    plt.title(mensagem_resultado_final)

    lista_custos[0] = lista_custos[1]

    #subplot = figure.add_subplot(1, 3, 1)
    plt.xlabel('Época')
    plt.ylabel('Custo')
    plt.plot(lista_epocas, lista_custos, 'blue')

    print('Tempo total: ' + str(datetime.datetime.now() - inicio))

    plt.show()

    print('Taxa de acerto do treino: ' + str(acuracia_treino))
    print('Taxa de acerto do teste: ' + str(acuracia_teste))

