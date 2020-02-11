#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy import misc, ndimage
import pylab as pl
import numpy as np
import cv2 as cv
import math




def histograma(imagen):
	#Se considera imagen en escala de grises
	h = np.zeros((256), dtype=np.int)
	data = np.zeros((256), dtype=np.int)
	for linea in imagen:
		for pixel in linea:
			
			h[pixel[0]]= h[pixel[0]] + 1

	return h

def del_fondo(imagen, n):
	
	for linea in imagen:
		for pixel in linea:
			if pixel[0] > n:
				pixel[0:] = 255


def contraste(imagen, minimo, maximo):
	#Formula (pixel - pixelMenor) * (255/pixelMayor - pixelMenor)
	#donde pixel mayor es igual al umbral 
	diferencia = float(255) / float(maximo - minimo)
	for linea in imagen:
		for pixel in linea:
			if (pixel[0] < maximo):
				if (pixel[0] - minimo < 0 ):
					p = float(0)
				else:
					p = float(pixel[0] - minimo)

				pixel[0:] = int ( p * diferencia)
				
def umbral_fondo(h, im):	
	A = 0.0 
	B = 0.0
	for g in range(0,256):
		
		A = A + (g * h[g])
		B = B + (g**2 * h[g])
		
	N = float(len(im) * len(im[0]))
	media = 1/N * A
	#varianza = 1/N * (B - (1/N) * A**2)
	#desviacion = math.sqrt(varianza)
	#print (media, varianza, desviacion)
	return media

def ajuste_histograma(histograma, imagen, umbral):
	#suponemos que la region de interes del histograma se encuentra de 0-umbral
	#calculamos el valor maximo
	maximo = (0,0)
	total_pixeles = 0
	for i in range(0, umbral):
		if (histograma[i] > maximo[0]):
			maximo = (histograma[i], i)
		total_pixeles = total_pixeles + histograma[i]
	#Porcentaje de eliminacion de ruido
	corte = maximo[0] * 5 / 100
	#Calculamos los nuevos limites (izquierdo y derecho)
	limite_izq = 0
	limite_der = umbral 
	for a in range(maximo[1], 0, -1):
		if (histograma[a] < corte):
			limite_izq = a
			break
	for i in range(maximo[1], umbral):
		if (histograma[i] < corte):
			limite_der = i
			break
	
	'''
	Eliminar ruido de la imagen (Se elimina tambien el contorno de la mano)
	'''
	for linea in imagen:
		for pixel in linea:
			if(pixel[0] < umbral):
				if (pixel[0] >= limite_der):
					pixel[0:] = 255
				elif (pixel[0] <= limite_izq):
					pixel[0:] = 0
	return [limite_izq, limite_der]

def makeMeanTables(h, K):
	'''
	Devuelve dos arrays con las medianas del histograma y la sumatoria de todos sus valores
	Recibe un histograma en escala de grises y el tamaño del histograma.
	'''
	foreground = np.zeros(K-1,int)
	backgroud = np.zeros(K-1,int)
	n0 = 0
	s0 = 0
	for q in range(0,K-1):
		n0 = n0 + h[q]
		s0 = s0 + (q*h[q])
		if (n0 > 0):
			backgroud[q] = s0/n0
		else:
			backgroud[q] = -1
	N = n0
	n1 = 0
	s1 = 0
	for q in reversed(range(0, K-2)):
		n1 = n1 + h[q]
		s1 = s1 + (q+1)*h[q+1]
		if (n1 > 0):
			foreground[q] = s1/n1
		else:
			foreground[q] = -1

	return (backgroud,foreground, N)

def otsu(h):
	'''
	Input: histograma en escala de grises
	Return: el mejor threshold o -1 si no existe
	'''
	K = len(h)
	u0, u1, N = makeMeanTables(h, K)
	
	b_max = 0
	q_max = -1
	n0 = 0
	for q in range(0, K-2):
		
		n0 = n0 + h[q]
		n1 = N - n0
		if (n0 > 0) and (n1 >0):
			b = int ( (1/float(N*N)) * n0 * n1 * pow(u0[q] - u1[q], 2))
			
			if (b > b_max):
				b_max = b
				q_max = q
	return (q_max)

def sup_up(n, im):
	for linea in im:
		for pixel in linea:
			if (pixel[0] > n + 68):
				pixel[0:] = 255

def image_equalize(img):

	w = len(img[0])
	h = len(img)
	M = float( w * h)
	K = 255
	h = histograma(img)
	#compute the cumulative histogram:
	for j in range(1, K-1):
		h[j] = h[j-1] + h[j]
	M = M - h[255]
	#equalize the image:
	for linea in img:
		for pixel in linea:
			if (pixel[0] < 255):
				a = pixel[0]
				b = h[a] * (K-2) / M
				pixel[0:] = b
	#return h


#Programa principal ----------
def volumen():
	for i in range (1, 8):
		im = misc.imread('Test/'+str(i)+'.tif')	#se carga la imagen
		h = histograma(im)		#se genera el array del histograma
		#print ("media de misc ", im.mean(), "minimo de misc: ", im.min(), "maximo de misc: ", im.max())
		#minimo = min_histograma(h)
		copy = np.copy(im)		#se crea una copia de la imagen para poder comparar resultados 
		ndimage.median_filter(copy, 3)
		media = umbral_fondo(h, copy)
		del_fondo(copy, media)
		limites = ajuste_histograma(h, copy, int(media))
		contraste(copy, limites[0], limites[1])
		h1 = histograma(copy)
		misc.imsave('Test/Resultados/mediana_limites'+str(i)+'.tif',copy)
def programa():
	im = misc.imread('Test/2.tif')	#se carga la imagen
	misc.imsave('Test/Resultados/otsu3/original.tif',im)
	#h = image_equalize(im)
	#misc.imsave('Test/Resultados/otsu2/equalizer.tif',im)
	h = histograma(im)		#se genera el array del histograma
	#print ("media de misc ", im.mean(), "minimo de misc: ", im.min(), "maximo de misc: ", im.max())
	#minimo = min_histograma(h)
	copy = np.copy(im)		#se crea una copia de la imagen para poder comparar resultados 
	#copy = cv.bilateralFilter(im,9,15,15)
	#ndimage.median_filter(copy, 3)
	#media = umbral_fondo(h, copy)
	print ("Media: ", umbral_fondo(h,copy))
	media = otsu(h)
	print (media)
	if (media == -1):
		return 
	del_fondo(copy, media)
	misc.imsave('Test/Resultados/otsu3/otsu.tif',copy)
	limites = ajuste_histograma(h, copy, int(media))
	contraste(copy, limites[0], limites[1])
	misc.imsave('Test/Resultados/otsu3/otsu_contraste.tif',copy)
	bi = cv.bilateralFilter(copy,9,15,15)
	misc.imsave('Test/Resultados/otsu3/otsu_bilateral.tif',bi)
	#img_grey = cv.cvtColor(bi, cv.COLOR_BGR2GRAY)
	#misc.imsave('Test/Resultados/otsu_m.tif',copy)
	#th2 = cv.adaptiveThreshold(img_grey,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,101,7)
	h1 = histograma(bi)
	image_equalize(bi)
	h2 = histograma(bi)
	misc.imsave('Test/Resultados/otsu3/otsu_bilateral_equialize.tif',bi)
	#sup_up(umbral2, bi)
	#misc.imsave('Test/Resultados/otsu1/otsu_sup_2.tif',bi)
	#blur = cv2.bilateralFilter(copy,9,50,50)
	
	#canny = cv2.Canny(copy,115,135)
	
	#misc.imsave('Test/Resultados/otsu7.tif',th2)
	#misc.imsave('Test/Resultados/prebilateral.tif',blur)

	x = np.arange(0,256,1)	#se generan los valores 'x' de la grafica 
	x1 = np.arange(0,int(media),1)	#se generan los valores 'x' de la grafica 
	pl.subplot(312)
	pl.plot(x[:254],h1[:254])			#se grafica
	pl.grid()
	pl.subplot(313)
	pl.plot(x[:254],h2[:254])
	pl.grid()				#anexamos un grid a la grafica
	pl.subplot(321)
	pl.axis('off')
	pl.title('Original')
	pl.imshow(im)
	pl.subplot(322)
	pl.axis('off')
	pl.title('Resultado')
	pl.imshow(bi)
	pl.show()

programa()
#volumen()

#Aplicar suavizado inicial
#recorte de contraste dinamico 2%-3%