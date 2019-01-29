# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:09:35 2019

@author: super
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

#Funcion que escala la imagen
from skimage.transform import resize
def escala(img):
  return resize(img, (200, 200))

#Funcion que convierte una matriz en cuadrada
def haceCuadrada(matriz):
    m, n = matriz.shape
    mayor = max(m, n)
    cuadrada = np.zeros((mayor, mayor))
    for i in range(m):
        for j in range(n):
            cuadrada[i][j] = cuadrada[i][j] + matriz[i][j]
    return cuadrada
#Funcion que hace la convolucion de una imagen con un kernel
def convolucion(imagen, kernel):
    m, n = kernel.shape
    if (m != n):
        kernel = haceCuadrada(kernel)        
    if (m == n):
        y, x = imagen.shape
        y = y - m + 1
        x = x - m + 1
        convolucionada = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                convolucionada[i][j] = np.sum(imagen[i:i+m, j:j+m]*kernel)
        convolucionada[convolucionada <= 0] = 0
    return convolucionada

#Muestra la imagen original
img = io.imread('Lena-grayscale.jpg')
plt.figure()
io.imshow(img)
plt.axis('off')
plt.title('Imagen original')
plt.show()

#Muestra la imagen escalada a 200x200 px
img = escala(img)
plt.figure()
io.imshow(img)
plt.title('Imagen escalada a 200x200px')
plt.axis('off')
plt.show()
print('El tamaño de la imagen es: ', img.shape)
print('La intensidad minima es: ', img.min())
print('La intensidad maxima es: ', img.max())

#Mascara con valores booleanos
mascara = img > 0.8
copiaImg = np.zeros((200,200))
copiaImg[img > 0.8] = img[img > 0.8]
plt.figure()
io.imshow(copiaImg)
plt.axis('off')
plt.title('Imagen con pixeles cuya intensidad sea mayor a 0.8')
plt.show()

#Convolucion con kernel 3x3
k = np.array([[-1, -1, -1],[2, 2, 2],[-1, -1, -1]])
convo = convolucion(img, k)
plt.figure()
plt.imshow(convo, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 3x3')
plt.axis('off')
plt.show()

#Convolucion con kernel 3x3 transpuesto
k = k.transpose()
convo = convolucion(img, k)
plt.figure()
plt.imshow(convo, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 3x3 transpuesto')
plt.axis('off')
plt.show()

#Convolucion con kernel de 5x5 una vez
k = 1/256 * np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
convo = convolucion(img, k)
plt.figure()
plt.imshow(convo, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 5x5 una vez')
plt.axis('off')
plt.show()

#Convolucion con kernel de 5x5 cuatro veces consecutivas 
convo = convolucion(img, k)
convo = convolucion(convo, k)
convo = convolucion(convo, k)
convo = convolucion(convo, k)
plt.figure()
plt.imshow(convo, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 5x5 cuatro veces consecutivas')
plt.axis('off')
plt.show()



