import datetime

inicio = datetime.datetime.now()
print("Início: " + str(inicio))

import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

next_batch_index = 0
RESIZE_WIDTH = 64
RESIZE_HEIGHT = 128
REAL_IMAGE_PATH = 'input/real/'
RESIZED_IMAGE_PATH = 'input/resized/'

image_list = []
kernel = np.ones((5,5), np.uint8)

print("Lendo imagens...")

for filename in glob.glob(REAL_IMAGE_PATH + '*.*'):

    image = cv.imread(filename, cv.IMREAD_COLOR)

    # RGB 2 GRAY
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # CLOSING
    #image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    # EROSION
    image = cv.erode(image, kernel, iterations=1)

    # RESIZE
    image = cv.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

    cv.imwrite(RESIZED_IMAGE_PATH + filename.split('\\')[1], image)

    x = 0

print("Imagens lidas")
print("\n\n\n")

fim = datetime.datetime.now()
print("Fim: " + str(fim) + '\n')

print("Duração: " + str(fim - inicio))