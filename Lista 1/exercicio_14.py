# Importamos os módulos do NumPy e do TensorFlow, além de dar um alias a cada um
import numpy as np
import tensorflow as tf

# Iniciamos a sessão do TensorFlow
sess = tf.Session()

# Definimos uma função print_tf, que imprime o tipo e o valor de cada entrada
def print_tf(x):
    print("TIPO: \n %s" % (type(x)))
    print("Valor: \n %s" % (x))
hello = tf.constant("www.deeplearningbrasil.com.br")
print_tf(hello)

# Iniciamos uma sessão que recebe como parâmetro a constante "deeplearningbrasil", além de imprimir tipo e valor da mesma
hello_out = sess.run(hello)
print_tf(hello_out)

# Definimos as constantes "1.5" e "2.5", além de imprimir tipo e valor de ambas
a = tf.constant(1.5)
b = tf.constant(2.5)
print_tf(a)
print_tf(b)

# Iniciamos sessões que recebem como parâmetros as constantes anteriores, além de imprimir tipo e valor de ambas
a_out = sess.run(a)
b_out = sess.run(b)
print_tf(a_out)
print_tf(b_out)

# Definimos uma operação de soma das constantes via TensorFlow, além de imprimir tipo e valor da mesma
a_plus_b = tf.add(a, b)
print_tf(a_plus_b)

# Iniciamos uma sessão que recebe a soma anterior como parâmetro, além de imprimir tipo e valor da mesma
a_plus_b_out = sess.run(a_plus_b)
print_tf(a_plus_b_out)

# Definimos uma operação de multiplicação das constantes via TensorFlow, executamos uma sessão que recebe a multiplicação como parâmetro, além de imprimir tipo e valor da mesma
# Tivemos que comentar a linha de multiplicação, haja visto que a operação "mul" foi removida do TensorFlow, e substituída por "multiply"
#a_mul_b = tf.mul(a, b)
a_mul_b = tf.multiply(a, b)
a_mul_b_out = sess.run(a_mul_b)
print_tf(a_mul_b_out)

# Definimos uma matriz 5x2 de valores aleatórios, além de imprimir tipo e valor da mesma.
weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
print_tf(weight)

# Como a variável foi instanciada sem valor inicial, o TensorFlow irá acusar erro caso tentemos trabalhar com a variável dessa maneira
#weight_out = sess.run(weight)
#print_tf(weight_out)

# Nesse momento o TensorFlow irá inicializar todas as variáveis que ainda não foram iniciadas, como a "weight" era anteriormente.
# Após a correção do erro e a inicialização das variáveis, agora é possível trabalhar com a variável previamente criada
init = tf.initialize_all_variables()
sess.run(init)

# O TensorFlow irá executar a sessão, além de imprimir o tipo e o valor da matriz e de seus elementos internos, como também um aviso de que todas as variáveis foram inicializadas
weight_out = sess.run(weight)
print_tf(weight_out)
print ("INITIALIZING ALL VARIABLES")

# Definimos um placeholder do tipo Float, além de imprimir o tipo e o valor do placeholder
x = tf.placeholder(tf.float32, [None, 5])
print_tf(x)

# Definimos uma multiplicação da matriz "weight" com o placeholder "x", além de imprimir o tipo e o valor da operação
oper = tf.matmul(x, weight)
print_tf(oper)

# Definimos uma matriz 1x5, que é inserida na operação "oper" para ser multiplicada pela matriz "weight", além de imprimir o tipo e o valor dessa multiplicação
data = np.random.rand(1, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)

# Definimos uma matriz 2x5, que é inserida na operação "oper" para ser multiplicada pela matriz "weight", além de imprimir o tipo e o valor dessa multiplicação
data = np.random.rand(2, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)