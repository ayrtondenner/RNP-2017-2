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
        
    return accuracy_list

def exibe_imagens(array_train_images):
    figure = plt.figure()

    for i in range(6):
        subplot = figure.add_subplot(2, 3, i + 1)
        imagem = array_train_images[i]
        array_imagem = np.array([imagem])
        array_imagem = array_imagem.reshape(len(array_imagem), 3, 32, 32)
        array_imagem = array_imagem.transpose(0, 2, 3, 1)
        plt.imshow(array_imagem[0])

    plt.show()

cifar_array = cifar_array()

array_train_images = []
array_train_labels = []
array_test_images = []
array_test_labels = []

for cifar_batch in cifar_array:

    for image in cifar_batch[b'data']:
        image = image/255.0
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


#exibe_imagens(array_train_images)


TAXA_APRENDIZADO = 0.001
EPOCAS = 200
BATCH = 512
L2_BETA = 0.0001
KEEP_PROB_TRAIN = 0.5
KEEP_PROB_TEST = 1.0
VIA_TENSORBOARD = True
LOGDIR = '/tmp/tensorflow/cifar10-5'

x = tf.placeholder(tf.float32, shape=[None, 3072]) # Tamanho total de uma imagem 32 x 32 x 3
x_reshape = tf.reshape(x, [-1, 3, 32, 32])
x_reshape = tf.transpose(x_reshape, [0, 2, 3, 1])
y_real = tf.placeholder(tf.float32, shape=[None, 10]) # Total de 10 classes no CIFAR-10

# LAYER 1 - Convolutional, ReLU, Convolutional, ReLU and Pooling

kernel = 64

with tf.name_scope('Layer_1'):
    W1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, kernel], stddev=0.1), name='W1_1') # 32 features de tamanho 5 x 5
    b1_1 = tf.Variable(tf.constant(0.1, shape=[kernel]), name='b1_1') # bias para as 32 features

    L1 = tf.nn.conv2d(x_reshape, W1_1, strides=[1, 1, 1, 1], padding='SAME', name='L1_1_conv2d') + b1_1
    L1 = tf.nn.relu(L1, name='L1_1_relu')

    W1_2 = tf.Variable(tf.truncated_normal([3, 3, kernel, kernel], stddev=0.1), name='W1_2') # 32 features de tamanho 5 x 5
    b1_2 = tf.Variable(tf.constant(0.1, shape=[kernel]), name='b1_2') # bias para as 32 features

    L1 = tf.nn.conv2d(L1, W1_2, strides=[1, 1, 1, 1], padding='SAME', name='L1_2_conv2d') + b1_2
    L1 = tf.nn.relu(L1, name='L1_2_conv2d')

    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='L1_maxpool')

    L1 = tf.nn.lrn(L1, 4, bias=1, alpha=0.001 / 9, beta=0.75, name='L1_local_normalization')

    #L1 = tf.nn.dropout(L1, keep_prob=KEEP_PROB_TRAIN)

# LAYER 2 - Convolutional, ReLU, Convolutional, ReLU and Pooling
with tf.name_scope('Layer_2'):
    W2_1 = tf.Variable(tf.truncated_normal([3, 3, kernel, kernel * 2], stddev=0.1), name='W2_1') # 64 features de tamanho 5 x 5
    b2_1 = tf.Variable(tf.constant(0.1, shape=[kernel * 2]), name='b2_1') # bias para as 64 features

    L2 = tf.nn.conv2d(L1, W2_1, strides=[1, 1, 1, 1], padding='SAME', name='L2_1_conv2d') + b2_1
    L2 = tf.nn.relu(L2, name='L2_1_relu')

    W2_2 = tf.Variable(tf.truncated_normal([3, 3, kernel * 2, kernel * 2], stddev=0.1), name='W2_2') # 64 features de tamanho 5 x 5
    b2_2 = tf.Variable(tf.constant(0.1, shape=[kernel * 2]), name='b2_2') # bias para as 64 features

    L2 = tf.nn.conv2d(L2, W2_2, strides=[1, 1, 1, 1], padding='SAME', name='L2_2_conv2d') + b2_2
    L2 = tf.nn.relu(L2, name='L2_2_relu')

    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='L2_maxpool')

    L2 = tf.nn.lrn(L2, 4, bias=1, alpha=0.001 / 9, beta=0.75, name='L2_local_normalization')

#L2 = tf.nn.dropout(pool2, keep_prob=KEEP_PROB_TRAIN)

# LAYER 3 - Convolutional, ReLU, Convolutional, ReLU and Pooling
with tf.name_scope('Layer_3'):
    W3_1 = tf.Variable(tf.truncated_normal([3, 3, kernel * 2, kernel * 4], stddev=0.1), name='W3_1') # 64 features de tamanho 5 x 5
    b3_1 = tf.Variable(tf.constant(0.1, shape=[kernel * 4]), name='b3_1') # bias para as 64 features

    L3 = tf.nn.conv2d(L2, W3_1, strides=[1, 1, 1, 1], padding='SAME', name='L3_1_conv2d') + b3_1
    L3 = tf.nn.relu(L3, name='L3_1_relu')

    W3_2 = tf.Variable(tf.truncated_normal([3, 3, kernel * 4, kernel * 4], stddev=0.1), name='W3_2') # 64 features de tamanho 5 x 5
    b3_2 = tf.Variable(tf.constant(0.1, shape=[kernel * 4]), name='b3_2') # bias para as 64 features

    L3 = tf.nn.conv2d(L3, W3_2, strides=[1, 1, 1, 1], padding='SAME', name='L3_2_conv2d') + b3_2
    L3 = tf.nn.relu(L3, name='L3_2_relu')

    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='L3_maxpool')

    L3 = tf.nn.lrn(L3, 4, bias=1, alpha=0.001 / 9, beta=0.75, name='L3_local_normalization')

    #L3 = tf.nn.dropout(L3, keep_prob=KEEP_PROB_TRAIN)

# LAYER 4 - Fully Connected Layer
with tf.name_scope('Layer_4'):
    Wfc1 = tf.Variable(tf.truncated_normal([4 * 4 * kernel * 4, 1024], stddev=0.1), name='Wfc1') # 64 features foram feitas, e o tamanho final do último layer fora 8 x 8
    bfc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='bfc1') # 32 x 32 x 3

    L4 = tf.reshape(L3, [-1, 4 * 4 * kernel * 4], name='L4_reshape')
    L4 = tf.matmul(L4, Wfc1, name='L4_matmul') + bfc1
    L4 = tf.nn.relu(L4, name='L4_relu')
    #L4 = tf.nn.dropout(L4, keep_prob=KEEP_PROB_TRAIN)

    #L4 = tf.nn.dropout(L4, keep_prob=KEEP_PROB_TRAIN)
with tf.name_scope('Layer_5'):
    Wfc2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name='Wfc2')
    bfc2 = tf.Variable(tf.constant(0.1, shape=[1024]), name='bfc2')

    L5 = tf.reshape(L4, [-1, 1024], name='L5_reshape')
    L5 = tf.matmul(L5, Wfc2, name='L5_matmul') + bfc2
    L5 = tf.nn.relu(L5, name='L5_relu')
    #L5 = tf.nn.dropout(L5, keep_prob=KEEP_PROB_TRAIN)

# LAYER 5 - Readout
with tf.name_scope('Layer_6'):
    Wrl = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='Wrl') # conectando os 3072 valores para as 10 classes do CIFAR-10
    brl = tf.Variable(tf.constant(0.1, shape=[10]), name='brl') # 10 classes

    y_calculado = tf.matmul(L5, Wrl, name='y_matmul') + brl

with tf.name_scope('cross_entropy'):
    softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=y_calculado)
    with tf.name_scope('total'):
        custo = tf.reduce_mean(softmax)

        regularizadores = (
            tf.nn.l2_loss(W1_1) + 
            tf.nn.l2_loss(W1_2) + 
            tf.nn.l2_loss(W2_1) +
            tf.nn.l2_loss(W2_2) +
            tf.nn.l2_loss(W3_1) +
            tf.nn.l2_loss(W3_2) +
            tf.nn.l2_loss(Wfc1) +
            tf.nn.l2_loss(Wfc2) +
            tf.nn.l2_loss(Wrl)
            )

        custo = tf.reduce_mean(custo + L2_BETA * regularizadores)

tf.summary.scalar('custo', custo)

with tf.name_scope('otimizador'):
    otimizador = tf.train.AdamOptimizer(learning_rate=TAXA_APRENDIZADO).minimize(custo)

with tf.name_scope('acuracia'):
    with tf.name_scope('predicao'):
        predicao = tf.equal(tf.arg_max(y_real, 1), tf.arg_max(y_calculado, 1))
    with tf.name_scope('acuracia'):
        acuracia = tf.reduce_mean(tf.cast(predicao, tf.float32))
tf.summary.scalar('acuracia', acuracia)

with tf.Session() as session:

    print("Início do treinamento")

    merge = tf.summary.merge_all()

    if (VIA_TENSORBOARD == True):
        train_writer = tf.summary.FileWriter(LOGDIR + '/train', session.graph)
        test_writer = tf.summary.FileWriter(LOGDIR + '/test')

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

        epoca_atual = i + 1

        for j in range(len(chunked_train_images)): # o tamanho do array de imagens e labels é o mesmo
            chunk_images = chunked_train_images[j]
            chunk_labels = chunked_train_labels[j]
                
            if (VIA_TENSORBOARD == True):
                cost_result, _ = session.run([merge, otimizador], feed_dict={x: chunk_images, y_real: chunk_labels})
                train_writer.add_summary(cost_result, epoca_atual)
            else:
                cost_result, _ = session.run([custo, otimizador], feed_dict={x: chunk_images, y_real: chunk_labels})
                cost_list.append(cost_result)

        lista_epocas.append(epoca_atual)
        mensagem_epoca = 'Época: ' + str(epoca_atual)
        if(VIA_TENSORBOARD == False):
            
            custo_atual = np.average(cost_list)
            lista_custos.append(custo_atual)
            
            mensagem_epoca = mensagem_epoca + ', custo: ' + str(custo_atual)
            
        #train_writer.add_summary(custo_atual, epoca_atual)
            
        # TODO: corrigir duplicidade em cálculo de custo por lotes
        if (i == 0 or (i + 1) % 5 == 0):
            if(VIA_TENSORBOARD == False):
                acuracia_treino = np.average(calcula_acuracia_treino(array_train_images, array_train_labels))
                acuracia_teste = np.average(calcula_acuracia_teste(array_test_images, array_test_labels))
                mensagem_epoca = mensagem_epoca + ', acurácia do treino: ' + str(acuracia_treino) + ', acurácia do teste: ' + str(acuracia_teste)
            else:
                test_cost_list = []

                chunked_test_images = list(chunked(array_test_images, BATCH))
                chunked_test_labels = list(chunked(array_test_labels, BATCH))

                for j in range(len(chunked_test_images)): # o tamanho do array de imagens e labels é o mesmo
                    test_chunk_images = chunked_test_images[j]
                    test_chunk_labels = chunked_test_labels[j]

                    if (VIA_TENSORBOARD == True):
                        test_cost_result, _ = session.run([merge, acuracia], feed_dict={x: test_chunk_images, y_real: test_chunk_labels})
                        test_writer.add_summary(test_cost_result, epoca_atual)
                    else:
                        test_cost_result, _ = session.run([custo, otimizador], feed_dict={x: test_chunk_images, y_real: test_chunk_labels})
                        test_cost_list.append(test_cost_result)        
            '''
            test_writer.add_summary(test_custo_atual, epoca_atual)
            '''

            #if(VIA_TENSORBOARD == False):
            #    test_custo_atual = np.average(test_cost_list)
        
        tempo_epoca = datetime.datetime.now() - inicio_epoca
        mensagem_epoca = mensagem_epoca + '\nTempo de treino: ' + str(tempo_epoca) + '\n'
        
        print(mensagem_epoca)
        
        i += 1

    # RESULTADO DAS ACURÁCIAS
    
    acuracia_treino = np.average(calcula_acuracia_treino(array_train_images, array_train_labels))
    acuracia_teste = np.average(calcula_acuracia_teste(array_test_images, array_test_labels))

    print('Tempo total: ' + str(datetime.datetime.now() - inicio))
    mensagem_resultado_final = 'Taxa de aprendizado: ' + str(TAXA_APRENDIZADO) + ', batch: ' + str(BATCH)

    if(VIA_TENSORBOARD == False): 
        mensagem_resultado_final = mensagem_resultado_final + ', custo final: ' + str(lista_custos[-1])
        
    mensagem_resultado_final = mensagem_resultado_final  + ', acurácia do treino: ' + str(acuracia_treino) + ', acurácia do teste: ' + str(acuracia_teste)
    print(mensagem_resultado_final)
    if(VIA_TENSORBOARD == False):
        plt.title(mensagem_resultado_final)

        lista_custos[0] = lista_custos[1]

        #subplot = figure.add_subplot(1, 3, 1)
        plt.xlabel('Época')
        plt.ylabel('Custo')
        plt.plot(lista_epocas, lista_custos, 'blue')
        '''
        '''
        plt.show()

    print('Taxa de acerto do treino: ' + str(acuracia_treino))
    print('Taxa de acerto do teste: ' + str(acuracia_teste))
    
