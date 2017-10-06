from scipy import misc, ndimage
import pylab as pl
import numpy as np
import csv 
import cv2
import math
from PIL import Image
#Cargamos archivo de la imagen
def valores(lista):
	maximo = lista[0][0]
	minimo = lista[0][0]
	for i in lista:
		for j in i:
			if j > maximo:
				maximo = j
			elif j < minimo:
				minimo = j	
	return maximo, minimo

def histograma(lista):
	valores = dict()
	for i in lista:
		for j in i:
			if (valores.has_key(str(j))):
				valores[str(j)] = valores[str(j)] + 1
			else:
				valores[str(j)] = 1

	#Ordenamos los valores
	l = valores.items()
	l.sort()
	#devolvemos una lista de tuplas ordenadas 
	return l
def divisor_arrays(lista, divisor):
	#la lista contendra tuplas
	menor = []
	mayor = []
	for i in lista:
		if float(i[0]) < divisor:
			menor.append(i)
		else:
			mayor.append(i)
	return menor, mayor

def medidas_estadisticas(lista):	
	A = 0.0 
	B = 0.0
	N = 0.0
	for g in lista:
		A = A + (float(g[0]) * float(g[1]))
		B = B + (float(g[0])**2 * float(g[1]))
		N = N + float(g[1])

	#N = float(len(lista) * len(lista[0]))
	media = 1/N * A
	desviacion = math.sqrt(1/N * (B - (1/N) * A**2))
	print "media y desviaicon: ", media, desviacion
	return media, desviacion



temperatura = []
#leemos el arvhico de temperaturas
with open('Test2/3.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for linea in reader:
        l = []
        for row in linea:
        	if (row != ' '):
        		l.append(float(row))
        temperatura.append(l)
maximo, minimo = valores(temperatura)


print "el total de datos es: ", len(temperatura) * len(temperatura[0])
h = histograma(temperatura)

media, desviacion= medidas_estadisticas(h)
inicio, fin = divisor_arrays(h, media)
#calculamos las medidas de tendencia y desviacion del arreglo superiror a la media
media_sup, desviacion_sup = medidas_estadisticas(fin)
#calculamos los limites inferior y superiror de la media en base a la desviacion
limite_inferior = media_sup - 0.1
limite_superior = media_sup + 0.1

#Asignamos los valores ancho y largo de la imagen
w, h = len(temperatura), len(temperatura[0])
#Creamos la imagen con valores de 0
data = np.zeros((w, h, 3), dtype=np.uint8)
#recorremos el archivo de temperaturas y la imagen al mismo tiempo
for i in range(0, len(temperatura)):
	for j in range(0, len(temperatura[0])):
		if (temperatura[i][j] > media_sup and temperatura[i][j] < limite_superior):
			m = maximo -minimo
			valor_pixel =int((temperatura[i][j] - minimo) * 255 / m)
			print temperatura[i][j]
			#print valor_pixel			
			data[i][j]= valor_pixel
			
img = Image.fromarray(data, 'RGB')
print type(img)
# Detectamos los bordes con Canny
canny = cv2.Canny(data, 50, 150)
misc.imsave('Test2/Resultados/canny'+str(i)+'.bmp',canny)
cv2.imshow("canny", canny)
img.show()
