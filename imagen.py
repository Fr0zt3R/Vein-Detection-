from scipy import misc, ndimage
import pylab as pl
import numpy as np
import csv 
import cv2

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
				
def umbral_fondo(h, im):	
	A = 0.0 
	B = 0.0
	for g in range(0,256):
		
		A = A + (g * h[g])
		B = B + (g**2 * h[g])
		
	N = float(len(im) * len(im[0]))
	media = 1/N * A
	#varianza = 1/N * (B - (1/N) * A**2)
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
	corte = maximo[0] * 2 / 100
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
	#print "limite derecho: ", limite_der, "limite_izq: ", limite_izq

	for linea in imagen:
		for pixel in linea:
			#if(pixel[0] < umbral):
			if (pixel[0] >= limite_der):
				pixel[0:] = 255
			elif (pixel[0] <= limite_izq):
				pixel[0:] = 0
	return [limite_izq, limite_der]
class Formatter(object):
	    def __init__(self, im):
	        self.im = im
	        
	    def __call__(self, x, y):
	    	x = int(x)
	    	y = int(y)
	        t =  temperatura[y][x]
	        return 'x={:d}, y={:d}, temp={:.01f}, z='.format(x, y, t)




#Programa principal ----------
def volumen():
	for i in range (1, 8):
		im = misc.imread('Test2/'+str(i)+'.tif')	#se carga la imagen
		h = histograma(im)		#se genera el array del histograma
		copy = cv2.GaussianBlur(im, (5,5), 0)
		media = umbral_fondo(h, copy)
		#del_fondo(copy, media)
		limites = ajuste_histograma(h, copy, int(media))
		contraste(copy, limites[0], limites[1])
		h1 = histograma(copy)
		misc.imsave('Test2/Resultados/procesada'+str(i)+'.bmp',copy)
def programa():
	#Cargamos imagen
	im = misc.imread('Test2/7.tif')	
	#Cargamos archivo de la imagen
	with open('Test2/7.csv', 'r') as f:
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
	media = umbral_fondo(h, copy)
	#del_fondo(copy, media)
	limites = ajuste_histograma(h, copy, int(media))
	contraste(copy, limites[0], limites[1])
	h1 = histograma(copy)

	# Detectamos los bordes con Canny
	canny = cv2.Canny(copy, 50, 150)

	cv2.imshow("canny", canny)
	#misc.imsave('Test/Resultados/gaussian_limites2_pre.tif',copy)
	x = np.arange(0,256,1)	#se generan los valores 'x' de la grafica 
	x1 = np.arange(0,int(media),1)	#se generan los valores 'x' de la grafica 
	pl.subplot(312)
	pl.plot(x[:254],h[:254])			#se grafica
	pl.grid()
	pl.subplot(313)
	pl.plot(ax[1:],h2)
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

#hacer analisis de temperaturas y sacar medida de tendensia central con desviacion estandar 
#Obtener valores alojados en el area y aislarlos 