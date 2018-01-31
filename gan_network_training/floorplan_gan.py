import datetime

inicio = datetime.datetime.now()
print("Início: " + str(inicio))

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cv2

#matplotlib inline

import glob

def get_all_images(images_path):

    image_list = []

    print("Lendo imagens...")

    for filename in glob.glob(images_path + '*.*'):
        '''
        image = Image.open(filename)
        image.load()
        data = np.asarray(image, dtype="int32")
        image_list.append(data)
        '''

        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        array = np.array(image)
        array = array.flatten()
        #array = np.hstack(array)
        

        image_list.append(array)

        #if len(image_list) == 16:
        #    break

        #image_list.append(image)

    print("Imagens lidas")
    return np.array(image_list)

def floorplan_next_batch(batch_size, list_images):

    global next_batch_index

    batch_images = []

    for i in range(batch_size):

        batch_images.append(list_images[next_batch_index])

        next_batch_index += 1

        # Caso o index de próximo item seja maior que a quantidade de itens na lista, volta-se para a imagem inicial
        if(next_batch_index >= len(list_images)):
            next_batch_index = 0        

    return np.array(batch_images)
        

def generator(z_placeholder, batch_size, z_dim):

    TAMANHO_IMAGEM = (RESIZE_WIDTH * 2) * (RESIZE_HEIGHT * 2) * 1

    # Camada 1

    g_w1 = tf.get_variable('g_w1', [z_dim, TAMANHO_IMAGEM], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
    g_b1 = tf.get_variable('g_b1', [TAMANHO_IMAGEM], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))

    g1 = tf.matmul(z_placeholder, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, RESIZE_HEIGHT * 2, RESIZE_WIDTH * 2, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Camada 2

    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))

    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME') + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [RESIZE_HEIGHT * 2, RESIZE_WIDTH * 2])

    # Camada 3 

    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))

    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME') + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [RESIZE_HEIGHT * 2, RESIZE_WIDTH * 2])

    # Camada 4

    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
    g_b4 = tf.get_variable('g_b4', [1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))

    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME') + g_b4
    g4 = tf.sigmoid(g4)

    return g4


def discriminator(images, reuse=None):

    KERNEL = 32

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:

        # Camada 1 - Primeira camada convolucional
        # Retorna 32 filtros de tamanho 5 x 5
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, KERNEL], initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
        d_b1 = tf.get_variable('d_b1', [KERNEL], initializer=tf.constant_initializer(0))

        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME') + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.max_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        KERNEL = KERNEL * 2

        # Camada 2 - Segunda camada convolucional
        # Retorna 64 filtros de tamanho 5 x 5
        d_w2 = tf.get_variable('d_w2', [5, 5, KERNEL/2, KERNEL], initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
        d_b2 = tf.get_variable('d_b2', [KERNEL], initializer=tf.constant_initializer(0))

        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME') + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.max_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Camada 3 - Primeira camada fully connected

        entrada_fc = int((RESIZE_WIDTH / 4) * (RESIZE_HEIGHT / 4) * KERNEL)

        d_w3 = tf.get_variable('d_w3', [entrada_fc, 512], initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
        d_b3 = tf.get_variable('d_b3', [512], initializer=tf.constant_initializer(0))

        d3 = tf.reshape(d2, [-1, entrada_fc])
        d3 = tf.matmul(d3, d_w3) + d_b3
        d3 = tf.nn.relu(d3)

        # Camada 4 - Segunda camada fully connected - Readout

        d_w4 = tf.get_variable('d_w4', [512, 1], initializer=tf.truncated_normal_initializer(stddev=desvio_padrao))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))

        d4 = tf.matmul(d3, d_w4) + d_b4

        return d4

next_batch_index = 0
RESIZE_WIDTH = 64
RESIZE_HEIGHT = 128

desvio_padrao = 0.02
TAXA_TREINAMENTO = 10**(-4) # 0.0001
LOGDIR = '/tensorboard/floorplan_gan_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
IMAGES_PATH = 'input/resized/'

print("Logdir: " + LOGDIR)

image_list = get_all_images(IMAGES_PATH)

# Definindo batch de ruído aleatório
z_dimensions = 128
batch_size = 16

print("Criação de placeholders")

# Placeholder para ruído
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')

# Placeholder para imagens enviadas ao discriminador
x_placeholder = tf.placeholder(tf.float32, shape=[None, RESIZE_HEIGHT, RESIZE_WIDTH, 1], name='x_placeholder')

print("Criação das redes")

# Gerador de ruídos
Gz = generator(z_placeholder, batch_size, z_dimensions)

# Discriminador com imagens reais
Dx = discriminator(x_placeholder)

# Discriminador com imagens falsas
Dg = discriminator(Gz, reuse = True)

print("Criação das funções de custo")

# Loss de imagens reais
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))

# Loss de imagens falsas
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

# Loss para o gerador, que quer que suas imagens se passem por reais
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

print("Separação de variáveis por nome")

# Separando variáveis entre discriminador e gerador
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

#print(d_vars)
#print(g_vars)

print("Criação de optimizadores")

# Optimizador para o treino de discriminação de imagens falsas
d_trainer_fake = tf.train.AdamOptimizer(TAXA_TREINAMENTO).minimize(d_loss_fake, var_list=d_vars)

# Optimizador para o treino de discriminação de imagens reais
d_trainer_real = tf.train.AdamOptimizer(TAXA_TREINAMENTO).minimize(d_loss_real, var_list=d_vars)

# Optimizador para o treino de geração de imagens
g_trainer = tf.train.AdamOptimizer(TAXA_TREINAMENTO).minimize(g_loss, var_list=g_vars)

print("Início da sessão")

tf.get_variable_scope().reuse_variables()

with tf.Session() as sess:

    print("Criação de valores escalares")

    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
    tf.summary.scalar('Discriminator_loss_real', d_loss_real)

    print("Criação de rede geradora e escalar para imagem falsa")

    images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
    #real_image_batch = floorplan_next_batch(image_list, batch_size)

    tf.summary.image('Generated_images', images_for_tensorboard, 4)
    #tf.summary.image('Real_images', real_image_batch, batch_size)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)

    sess.run(tf.global_variables_initializer())

    print("Início do pré-treino")

    # Pré-treino do discriminador
    for i in range(300):

        z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
        real_image_batch = floorplan_next_batch(batch_size, image_list).reshape([batch_size, RESIZE_HEIGHT, RESIZE_WIDTH, 1])
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], {x_placeholder: real_image_batch, z_placeholder: z_batch})

        if (i % 100 == 0):
            print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

    # Treino do gerador e do discriminador

    print("Fim do pré-treino e início do treino real")

    for i in range(10**6):
        #print("Época #" + str(i))
        real_image_batch = floorplan_next_batch(batch_size, image_list).reshape([batch_size, RESIZE_HEIGHT, RESIZE_WIDTH, 1])
        z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])

        # Treino do discriminador com imagens reais e falsas
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], {x_placeholder: real_image_batch, z_placeholder: z_batch})

        # Treino do gerador

        z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
        _ = sess.run([g_trainer, g_loss], {z_placeholder: z_batch})

        if i % 10 == 0:
            z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
            summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
            writer.add_summary(summary, i)



print("Fim da execução")

print("\n\n\n")

fim = datetime.datetime.now()
print("Fim: " + str(fim) + '\n')

print("Duração: " + str(fim - inicio))