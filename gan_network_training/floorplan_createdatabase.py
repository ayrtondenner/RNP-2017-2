import datetime

inicio = datetime.datetime.now()
print("Início: " + str(inicio))

import shutil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import codecs
import subprocess

import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# your image path
#img_dir = '../DeepBuilding/input/images_sample/'
img_dir = '../DeepBuilding/input/Kaggle-renthop~/images/'

copy_dir = '../DeepBuilding/input/real/'

def white_count(img_url):
    im = Image.open(img_url)  
    w, h = im.size  
    colors = im.getcolors(w*h)
    return im.getcolors(w*h)[0][0] / (w * h * 1.0)

for listing_dir in os.listdir(img_dir):
    if len(listing_dir) != 7 or not listing_dir.isdigit():
        continue
    
    for listing_img in os.listdir(img_dir + listing_dir):
        if listing_img == '.DS_Store' or listing_img == listing_dir:
            continue
        img_full_dir = img_dir + listing_dir + '/' + listing_img
        white_scale = white_count(img_full_dir)
        if white_scale > 0.6:
            print(listing_img, white_scale)
            shutil.copy(img_full_dir, copy_dir)

print("\n\n\n")

fim = datetime.datetime.now()
print("Fim: " + str(fim) + '\n')

print("Duração: " + str(fim - inicio))