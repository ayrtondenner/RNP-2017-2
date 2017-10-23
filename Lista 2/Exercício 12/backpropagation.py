import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Neuronio:

    nome = None
    bias = 0
    camada = 0
    pesos_entrada = []
    pesos_saida = []

    def __init__(self, nome, bias, camada, pesos_entrada, pesos_saida):
        self.nome = nome
        self.bias = bias
        self.camada = camada
        self.pesos_entrada = pesos_entrada
        self.pesos_saida = pesos_saida

    def ativar_neuronio(self):

        vetor_entrada = []

        for peso in self.pesos_entrada:
            valor_peso = peso.valor
            valor_pai = peso.retorna_valor_pai()
            vetor_entrada.append([valor_peso, valor_pai])

        soma_ponderada = self.__soma_ponderada(vetor_entrada)
        sigmoide = self.__sigmoide(soma_ponderada)

        return sigmoide

    def __soma_ponderada(self, vetor_entrada):

        soma = 0

        for entrada in vetor_entrada:

            peso = entrada[0]
            valor = entrada[1]

            soma += peso * valor

        soma += self.bias

        return soma

    def __sigmoide(self, soma_ponderada):

        return 1/(1 + np.exp(-soma_ponderada))


class Peso:

    nome = None
    valor = 0
    neuronio_origem = None
    neuronio_destino = None
    valor_entrada = None

    def __init__(self, nome, valor, neuronio_origem, neuronio_destino, valor_entrada):
        self.nome = nome
        self.valor = valor
        self.neuronio_origem = neuronio_origem
        self.neuronio_destino = neuronio_destino
        self.valor_entrada = valor_entrada

        if self.neuronio_origem is not None:
            self.neuronio_origem.pesos_saida.append(self)

        if self.neuronio_destino is not None:
            self.neuronio_destino.pesos_entrada.append(self)
        else:
            return 0

def atualizar_peso(valor_peso, taxa_aprendizado, valor_derivada):

    return valor_peso - taxa_aprendizado * valor_derivada

def imprimir_grafico(lista_erro_por_epoca, epoca):
    x = []
    y = []

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for erro in lista_erro_por_epoca:
        x.append(erro[0])
        y.append(erro[1])

    ax.plot(x, y, 'blue')

    ax.set_title('Gráfico de erro com #{} épocas'.format(epoca))
    ax.set_xlabel('Nº de épocas')
    ax.set_ylabel('Erro total')

    plt.show()

taxa_aprendizado = 0.5

entradas = np.array([0.05, 0.1])
saidas = np.array([0.01, 0.99])

neuronios = np.array([
    Neuronio("h1", 0.35, 1, [], []),
    Neuronio("h2", 0.35, 1, [], []),
    Neuronio("o1", 0.6, 2, [], []),
    Neuronio("o2", 0.6, 2, [], [])
])

pesos = np.array([
    Peso("w1", 0.15, None, neuronios[0], entradas[0]),
    Peso("w2", 0.2, None, neuronios[0], entradas[1]),
    Peso("w3", 0.25, None, neuronios[1], entradas[0]),
    Peso("w4", 0.3, None, neuronios[1], entradas[1]),
    Peso("w5", 0.4, neuronios[0], neuronios[2], None),
    Peso("w6", 0.45, neuronios[1], neuronios[2], None),
    Peso("w7", 0.5, neuronios[0], neuronios[3], None),
    Peso("w8", 0.55, neuronios[1], neuronios[3], None)
])

epocas_max = 10001 # O último é não-incluso
epocas_imprimir = [50, 100, 1000, 5000, 10000]
lista_erro_por_epoca = []

for i in range(1, epocas_max):

    if i in epocas_imprimir:
        imprimir_grafico(lista_erro_por_epoca, i)

    resultados = np.array([neuronios[2].ativar_neuronio(), neuronios[3].ativar_neuronio()])
    erro_total = np.sum( (saidas - resultados)**2/2 )

    lista_erro_por_epoca.append([i, erro_total])

    novos_pesos = []
    novos_bias = []

    # Calculando os pesos
    for peso in pesos:
        derivada = derivada_erro_para_peso(saidas, peso)
        novo_peso = atualizar_peso(peso.valor, taxa_aprendizado, derivada)

        novos_pesos.append([peso, novo_peso])

    # Calculando o bias
    for neuronio in neuronios:
        derivada = derivada_erro_para_bias(saidas, neuronio)
        novo_bias = atualizar_peso(neuronio.bias, taxa_aprendizado, derivada)

        novos_bias.append([neuronio, novo_bias])

    # Atualizando os pesos
    for novo_peso in novos_pesos:
        novo_peso[0].valor = novo_peso[1]

    # Atualizando os bias
    for novo_bias in novos_bias:
        novo_bias[0].bias = novo_bias[1]