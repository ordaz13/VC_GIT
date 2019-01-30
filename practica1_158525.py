# -*- coding: utf-8 -*-
"""
Carlos Octavio Ordaz Bernal
158525
Visión por Computadora
29 de enero de 2019
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
    if m != n:
        kernel = haceCuadrada(kernel)
    #No me funciona usar la siguiente linea, marca error en parametros ?
    #kernel = np.flip(kernel)
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    y, x = imagen.shape
    m = m//2
    n = n//2
    convolucionada = np.zeros((y,x))
    for i in range(m, y-m):
        for j in range(n, x-n):
            suma = 0
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    suma = suma + kernel[k][l] * imagen[i-m+k][j-n+l]
            convolucionada[i][j] = suma
    convolucionada[convolucionada <= 0] = 0
    return convolucionada

#Funcion que devuelve la cantidad de veces que se aplica el kernel de 5x5 para obtener std < 0.1
def cuantasVeces(imagen, kernel):
    convo = convolucion(imagen, kernel)
    cuenta = 0
    while np.std(convo) >= 0.1:
        convo = convolucion(convo, kernel)
        cuenta = cuenta + 1
        print(np.std(convo),'\t',cuenta)
    return cuenta

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
valor = img.max()*0.8
mascara = img > valor
copiaImg = np.zeros((200,200))
copiaImg[img > valor] = img[img > valor]
plt.figure()
io.imshow(copiaImg)
plt.axis('off')
plt.title('Imagen con pixeles cuya intensidad sea mayor a 0.8')
plt.show()

#Convolucion con kernel 3x3
k = np.array([[-1, -1, -1],[2, 2, 2],[-1, -1, -1]])
convo = convolucion(img, k)
#print('El tamaño de la imagen es: ', convo.shape)
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
convoUna = convolucion(img, k)
plt.figure()
plt.imshow(convoUna, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 5x5 una vez')
plt.axis('off')
plt.show()

#Convolucion con kernel de 5x5 cuatro veces consecutivas 
convo = convolucion(img, k)
convo = convolucion(convo, k)
convo = convolucion(convo, k)
convoCuatro = convolucion(convo, k)
plt.figure()
plt.imshow(convoCuatro, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 5x5 cuatro veces consecutivas')
plt.axis('off')
plt.show()

#Cantidad de veces que se aplica el kernel de 5x5 para obtener std < 0.1
#res = cuantasVeces(img, k)
#print('Se necesito aplicar ', res, ' veces la convolucion')
#En total fueron 3282 veces

#Convolucion con un kernel de 3x3 y la imagen ya con kernel Gaussiano
k = np.array([[-1, -1, -1],[2, 2, 2],[-1, -1, -1]])
convo = convolucion(convoUna, k)
plt.figure()
plt.imshow(convo, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 3x3 sobre la imagen convolucionada una vez')
plt.axis('off')
plt.show()

#Convolucion con un kernel de 3x3 y la imagen con cuatro kernel Gaussiano
convo = convolucion(convoCuatro, k)
plt.figure()
plt.imshow(convo, cmap=plt.cm.gray)
plt.title('Convolucion con kernel de 3x3 sobre la imagen convolucionada cuatro veces')
plt.axis('off')
plt.show()

#Resta de la imagen convolucionada cuatro veces con kernel 3x3 y la original
k = k.transpose()
convo = convolucion(img, k)
convo = convolucion(convo, k)
convo = convolucion(convo, k)
convo = convolucion(convo, k)
resta = convo - img
plt.figure()
plt.imshow(resta, cmap=plt.cm.gray)
plt.title('Resta de la imagen convolucionada cuatro veces con kernel 3x3 y la original')
plt.axis('off')
plt.show()