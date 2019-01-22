from scipy import misc, ndimage
import pylab as pl
import numpy as np
import csv 
import cv2
import math

temperatura = []


def histograma(imagen):
	#Se considera imagen en escala de grises
	h = np.zeros((256), dtype=np.int)
	#data = np.zeros((256), dtype=np.int)
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
	print "diferencia: ", diferencia
	print "minimo: ", minimo
	print "maximo: ", maximo

	for linea in imagen:
		for pixel in linea:
			if (pixel[0] < maximo):
				if (pixel[0] - minimo < 0 ):
					p = float(0)
				else:
					p = float(pixel[0] - minimo)

				pixel[0:] = int ( p * diferencia)
				
def estadisticas(h, im, x, y):	
	A = 0.0 
	B = 0.0
	for g in range(x, y):
		
		A = A + (g * h[g])
		B = B + (g**2 * h[g])
		
	N = float(len(im) * len(im[0]))
	if y == 255:
		N = N - h[255]
	media = 1/N * A
	desviacion = math.sqrt(1/N * (B - (1/N) * A**2))
	return media, desviacion

def ajuste_histograma(histograma, imagen, umbral, desviacion):
	#suponemos que la region de interes del histograma se encuentra de 0-umbral
	#calculamos el valor maximo
	maximo = (0,0)
	total_pixeles = 0
	for i in range(0, umbral):
		if (histograma[i] > maximo[0]):
			maximo = (histograma[i], i)
		total_pixeles = total_pixeles + histograma[i]
	corte = maximo[0] * 45 / 100
	#Calculamos los nuevos limites (izquierdo y derecho)
	limite_izq = 0
	limite_der = umbral - desviacion
	for a in range(maximo[1], 0, -1):
		if (histograma[a] < corte):
			limite_izq = a
			break
	#for i in range(maximo[1], umbral):
	#	if (histograma[i] < corte):
	#		limite_der = i
	#		break
	print "deviacion de original: ", umbral - desviacion
	print "limite derecho calculado: ", limite_der
	for linea in imagen:
		for pixel in linea:
			#if(pixel[0] < umbral):
			if (pixel[0] >= limite_der):
				pixel[0:] = 255
			elif (pixel[0] <= limite_izq):
				pixel[0:] = 255
	return [limite_izq, limite_der]

class Formatter(object):
	    def __init__(self, im):
	        self.im = im
	        
	    def __call__(self, x, y):
	    	x = int(x)
	    	y = int(y)
	        t =  temperatura[y][x]
	        return 'x={:d}, y={:d}, temp={:.01f}, z='.format(x, y, t)

def elim_sup(imagen, media, desviacion, histograma):
	limite = int(media + desviacion)
	limite_izq = int(media - desviacion)
	for x in imagen:
		for y in x:
			
			if y[0] != 255 and y[0] >limite:
				y[0:] = 255

	maximo = (0,0)
	total_pixeles = 0
	for i in range(0, limite):
		if (histograma[i] > maximo[0]):
			maximo = (histograma[i], i)
		total_pixeles = total_pixeles + histograma[i]
	corte = maximo[0] * 2 / 100

	for a in range(int(limite - desviacion), 0, -1):
		if (histograma[a] < corte):
			limite_izq = a
			break
	return [limite_izq, limite]


#Programa principal ----------
def volumen():
	for i in range (1, 8):
		im = misc.imread('Test2/'+str(i)+'.tif')	#se carga la imagen
		h = histograma(im)		#se genera el array del histograma
		copy = cv2.GaussianBlur(im, (5,5), 0)
		media, desviacion = estadisticas(h, copy, 0, 256)
		#del_fondo(copy, media)
		limites = ajuste_histograma(h, copy, int(media))
		contraste(copy, limites[0], limites[1])
		h1 = histograma(copy)
		misc.imsave('Test2/Resultados/procesada'+str(i)+'.bmp',copy)
def programa():
	#Cargamos imagen
	im = misc.imread('Test2/4.tif')	
	#Cargamos archivo de la imagen
	with open('Test2/4.csv', 'r') as f:
	    reader = csv.reader(f, delimiter=',')
	    for linea in reader:
	        l = []
	        for row in linea:
	        	if (row != ' '):
	        		l.append(float(row))
	        temperatura.append(l)
	print temperatura[0][1]
	h2, ax = np.histogram(temperatura, bins = 100)


	h = histograma(im)		#se genera el array del histograma

	#copy = np.copy(im)		#se crea una copia de la imagen para poder comparar resultados 
	copy = cv2.GaussianBlur(im, (5,5), 0)

	#ndimage.gaussian_filter(copy, 3)
	media, desviacion = estadisticas(h, copy, 0, 256)
	print "media: ",media, "Desviacion: ", desviacion, "Limite inferior: ", media-desviacion
	#del_fondo(copy, media)
	limites = ajuste_histograma(h, copy, int(media), desviacion)
	contraste(copy, limites[0], limites[1])
	h1 = histograma(copy)
	# Detectamos los bordes con Canny
	#canny = cv2.Canny(copy, 50, 150)

	#cv2.imshow("canny", canny)
	'''
	#Aqui insertamos metodo iterativo para cortar la imagen
	for i in range(0, 1):
		h1 = histograma(copy)
		media, desviacion = estadisticas(h1, copy,1, 255)
		print media, desviacion
		limites = elim_sup(copy, media, desviacion, h1)
		contraste(copy, limites[0], limites[1])
		h1 = histograma(copy)


		copy = cv2.GaussianBlur(copy, (5,5), 0)

	# Detectamos los bordes con Canny
	canny = cv2.Canny(copy, 50, 150)

	cv2.imshow("canny", canny)'''
	misc.imsave('Test2/Resultados/Nuevos/2_45_sc.bmp',copy)
	x = np.arange(0,256,1)	#se generan los valores 'x' de la grafica 
	x1 = np.arange(0,int(media),1)	#se generan los valores 'x' de la grafica 
	pl.subplot(312)
	pl.plot(x,h)			#se grafica
	pl.grid()
	pl.subplot(313)
	pl.plot(x[1:254],h1[1:254])
	pl.grid()				#anexamos un grid a la grafica
	ax = pl.subplot(321)
	pl.axis('off')
	pl.title('Original')
	img = pl.imshow(im)
	ax.format_coord = Formatter(img)
	ax1 = pl.subplot(322)
	pl.axis('off')
	pl.title('Resultado')
	img1 = pl.imshow(copy)
	ax1.format_coord = Formatter(img1)
	pl.show()

programa()

#Aplicar suavizado inicial
#recorte de contraste dinamico 2%-3%
#filtro winer
#hacer analisis de temperaturas y sacar medida de tendencia central con desviacion estandar 
#Obtener valores alojados en el area y aislarlos 