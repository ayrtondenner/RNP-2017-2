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
        array = array/255
        

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

    GEN_WIDHT = int(WIDTH/16)
    GEN_HEIGHT = int(HEIGHT/16)
    CH1, CH2, CH3, CH4, = 512, 256, 128, 64 #number of kernels/filters

    with tf.variable_scope('gen') as scope:

        # Camada 1 - Saída: 4 * 8 * 256

        g_w1 = tf.get_variable('g_w1', [z_dim, GEN_WIDHT * GEN_HEIGHT * CH1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO))
        g_b1 = tf.get_variable('g_b1', [GEN_WIDHT * GEN_HEIGHT * CH1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0))

        g1 = tf.add(tf.matmul(z_placeholder, g_w1), g_b1, name='flat_conv1')

        g1 = tf.reshape(g1, [-1, GEN_HEIGHT, GEN_WIDHT, CH1], name='reshape_1')
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, decay=0.9, scope='bn1')
        g1 = tf.nn.relu(g1, name='act_1')

        # Camada 2 - Saída: 8 * 16 * 128

        g2 = tf.layers.conv2d_transpose(g1, CH2, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='transpose_2')
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, decay=0.9, scope='bn2')
        g2 = tf.nn.relu(g2, name='act_2')

        # Camada 3 - Saída: 16 * 32 * 64

        g3 = tf.layers.conv2d_transpose(g2, CH3, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='transpose_3')
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, decay=0.9, scope='bn3')
        g3 = tf.nn.relu(g3, name='act_3')

        # Camada 3 - Saída: 32 * 64 * 32

        g4 = tf.layers.conv2d_transpose(g3, CH4, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='transpose_4')
        g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, decay=0.9, scope='bn4')
        g4 = tf.nn.relu(g4, name='act_4')

        # Camada 4 - Saída: 64 * 128 * 1

        g5 = tf.layers.conv2d_transpose(g4, CHANNEL, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='transpose_5')
        # g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, decay=0.9, scope='bn6')
        g5 = tf.nn.tanh(g5, name='act_5')

        return g5


def discriminator(images, reuse=None):

    CH1, CH2, CH3, CH4 = 64, 128, 256, 512

    with tf.variable_scope('dis') as scope:

        if reuse:
            scope.reuse_variables()

        # Camada 1 - Saída: 64 * 128 * 64

        d1 = tf.layers.conv2d(images, CH1, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='conv1')
        d1 = tf.contrib.layers.batch_norm(d1, epsilon=1e-5, decay=0.9, scope='bn1')
        d1 = tf.nn.relu(d1, name='act_1')

        # Camada 2 - Saída: 64 * 128 * 128

        d2 = tf.layers.conv2d(d1, CH2, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='conv2')
        d2 = tf.contrib.layers.batch_norm(d2, epsilon=1e-5, decay=0.9, scope='bn2')
        d2 = tf.nn.relu(d2, name='act_2')

        # Camada 3 - Saída: 64 * 128 * 256

        d3 = tf.layers.conv2d(d2, CH3, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='conv3')
        d3 = tf.contrib.layers.batch_norm(d3, epsilon=1e-5, decay=0.9, scope='bn3')
        d3 = tf.nn.relu(d3, name='act_3')

        # Camada 4 - Saída: 64 * 128 * 512

        d4 = tf.layers.conv2d(d3, CH4, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO), name='conv4')
        d4 = tf.contrib.layers.batch_norm(d4, epsilon=1e-5, decay=0.9, scope='bn4')
        d4 = tf.nn.relu(d4, name='act_4')

        #layer_shape = int(np.prod(d4.get_shape()[1:]))
        layer_shape = int(np.prod(d4.get_shape()[1:]))

        # First fully connected layer
        d_w5 = tf.get_variable('d_w5', [layer_shape, 512], initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO))
        d_b5 = tf.get_variable('d_b5', [512], initializer=tf.constant_initializer(0))
        d5 = tf.reshape(d4, [-1, layer_shape])
        d5 = tf.add(tf.matmul(d5, d_w5), d_b5, name='flat_conv1')
        d5 = tf.nn.relu(d5, name='d5_relu')

        # Second fully connected layer
        d_w6 = tf.get_variable('d_w6', [512, 1], initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO))
        d_b6 = tf.get_variable('d_b6', [1], initializer=tf.constant_initializer(0))
        d6 = tf.add(tf.matmul(d5, d_w6), d_b6, name='flat_conv2')
        # wgan just get rid of the sigmoid
        #g5 = tf.add(tf.matmul(d5, d_w5), d_b5, name='logits')
        # dcgan
        #acted_out = tf.nn.sigmoid(d5)
        return d6 #, acted_out
        #d5 = tf.nn.relu(d5, name='act_4')

        # Second fully connected layer
        #d_w5 = tf.get_variable('d_w5', [512, 1], initializer=tf.truncated_normal_initializer(stddev=DESVIO_PADRAO))
        #d_b5 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))
        #d5 = tf.add(tf.matmul(d4, d_w5), d_b5, name='flat_conv2')

        # d4 contains unscaled values
        #return d5


next_batch_index = 0

WIDTH, HEIGHT, CHANNEL = 64, 128, 1
DESVIO_PADRAO = 0.02
TAXA_TREINAMENTO = 10**(-4) # 0.0001

LOGDIR = '/tensorboard/floorplan_gan_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
IMAGES_PATH = 'input/resized/'

print("Logdir: " + LOGDIR)

image_list = get_all_images(IMAGES_PATH)

# Definindo batch de ruído aleatório
z_dimensions = 100
batch_size = 64

print("Criação de placeholders")

# Placeholder para ruído
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')

# Placeholder para imagens enviadas ao discriminador
x_placeholder = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 1], name='x_placeholder')

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
t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'dis' in var.name]
g_vars = [var for var in t_vars if 'gen' in var.name]

#print(d_vars)
#print(g_vars)

print("Criação de optimizadores")

# Optimizador para o treino de discriminação de imagens falsas
d_trainer_fake = tf.train.RMSPropOptimizer(TAXA_TREINAMENTO).minimize(d_loss_fake, var_list=d_vars)

# Optimizador para o treino de discriminação de imagens reais
d_trainer_real = tf.train.RMSPropOptimizer(TAXA_TREINAMENTO).minimize(d_loss_real, var_list=d_vars)

# Optimizador para o treino de geração de imagens
g_trainer = tf.train.RMSPropOptimizer(TAXA_TREINAMENTO).minimize(g_loss, var_list=g_vars)

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
        real_image_batch = floorplan_next_batch(batch_size, image_list).reshape([batch_size, HEIGHT, WIDTH, 1])
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], {x_placeholder: real_image_batch, z_placeholder: z_batch})

        if (i % 100 == 0):
            print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

    # Treino do gerador e do discriminador

    print("Fim do pré-treino e início do treino real")

    for i in range(10**6):
        #print("Época #" + str(i))
        real_image_batch = floorplan_next_batch(batch_size, image_list).reshape([batch_size, HEIGHT, WIDTH, 1])
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