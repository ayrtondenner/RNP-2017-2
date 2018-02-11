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
SHOW_FINAL_RESULT = False

image_list = []
kernel = np.ones((5,5), np.uint8)

print("Lendo imagens...")

for filepath in glob.glob(REAL_IMAGE_PATH + '*.*'):

    image = cv.imread(filepath, cv.IMREAD_COLOR)

    # RGB 2 GRAY
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # CLOSING
    #image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    # EROSION
    image = cv.erode(image, kernel, iterations=1)

    # RESIZE
    image = cv.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

    filename = filepath.split('\\')[1]

    cv.imwrite(RESIZED_IMAGE_PATH + filename, image)

    #saved_image = cv.imread(RESIZED_IMAGE_PATH + filename, cv.IMREAD_GRAYSCALE)

    #denoised_image = cv.fastNlMeansDenoising(saved_image, None, 10, 7, 21)

    #cv.imwrite(RESIZED_IMAGE_PATH + filename, denoised_image)

    #x = 0

print("Imagens lidas")
print("\n\n\n")

fim = datetime.datetime.now()
print("Fim: " + str(fim) + '\n')

print("Duração: " + str(fim - inicio))

if SHOW_FINAL_RESULT:

    for filepath in glob.glob(RESIZED_IMAGE_PATH + '*.*'):

        saved_image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        denoised_image = cv.fastNlMeansDenoising(saved_image, None, 10, 7, 21)

        figure = plt.figure(figsize=(2, 2))

        figure.add_subplot(1, 1, 1)
        plt.imshow(saved_image, 'gray')

        figure.add_subplot(1, 2, 2)
        plt.imshow(denoised_image, 'gray')

        plt.show()