import numpy as np
import matplotlib.pyplot as plt

def graph(array_pesos, epoca):

    # w1 * x + w2 * y + b = 0
    # x = (-b - w2 * y)/w1
    # y = (-b - w1 * x)/w2
    #x = [-array_pesos[2]/array_pesos[0], 0]
    #y = [0, -array_pesos[2]/array_pesos[1]]        

    x = []
    y = []

    for i in range(-1, 3):

        x.append(i)
        y.append( (-array_pesos[2] - array_pesos[0] * i)/array_pesos[1] )

    plt.plot(x, y, label="Epoca #{}".format(epoca))

def funcao_limite(valor):
    if valor >= 0:
        return 1
    elif valor < 0:
        return 0

def atualizar_pesos(treino, valor_previsto):

    for i in range(0, len(array_pesos)):

        if(i == len(array_pesos) - 1):
            paramentro = 1 # Parâmetro do bias
        else:            
            paramentro = treino[i] # Outros parâmetros do método "OR"
        
        array_pesos[i] = array_pesos[i] + taxa_aprendizado * (treino[2] - valor_previsto) * paramentro

base_treino_or = np.array([
    np.array([0, 0, 0]),
    np.array([0, 1, 1]),
    np.array([1, 0, 1]),
    np.array([1, 1, 1]),
])

array_pesos = np.array([0.3092, 0.3129, -0.8649])

taxa_aprendizado = 1

epocas = 0
numero_treinos_corretos = 0

graph(array_pesos, epocas)

while numero_treinos_corretos < len(base_treino_or):

    epocas += 1
    numero_treinos_corretos = 0

    for treino in base_treino_or:

        resultado = array_pesos[0] * treino[0] + array_pesos[1] * treino[1] + array_pesos[2] * 1

        resultado_limite = funcao_limite(resultado)

        plt.plot()

        if resultado_limite == treino[2]:
            numero_treinos_corretos += 1
        else:
            atualizar_pesos(treino, resultado_limite)
            graph(array_pesos, epocas)

print("Peso #1: " + str(array_pesos[0]))
print("Peso #2: " + str(array_pesos[1]))
print("Peso #3 (bias): " + str(array_pesos[2]))
print("Nº de épocas: " + str(epocas))

for treino in base_treino_or:

    plt.plot(treino[0], treino[1], 'bo' if treino[2] == 1 else 'ro')

plt.show()