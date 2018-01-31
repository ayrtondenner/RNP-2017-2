import datetime

inicio = datetime.datetime.now()
print("Início: " + str(inicio))

import glob
import cv2
#import matplotlib.pyplot as plt
#from PIL import Image

next_batch_index = 0
RESIZE_WIDTH = 64
RESIZE_HEIGHT = 128
REAL_IMAGE_PATH = 'input/real/'
RESIZED_IMAGE_PATH = 'input/resized/'

image_list = []

print("Lendo imagens...")

for filename in glob.glob(REAL_IMAGE_PATH + '*.*'):

    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))
    cv2.imwrite(RESIZED_IMAGE_PATH + filename.split('\\')[1], image)

    #image = Image.open(filename)
    #image = image.resize( (RESIZE_WIDTH, RESIZE_HEIGHT) )
    #image.save(RESIZED_IMAGE_PATH + filename.split('\\')[1])

print("Imagens lidas")
print("\n\n\n")

fim = datetime.datetime.now()
print("Fim: " + str(fim) + '\n')

print("Duração: " + str(fim - inicio))