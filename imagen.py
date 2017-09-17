from scipy import misc
from scipy import ndimage
import pylab as pl
import numpy as np




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
				
def umbral_fondo(histograma):	
	A = 0.0 
	B = 0.0
	for g in range(0,256):
		
		A = A + (g * h[g])
		B = B + (g**2 * h[g])
		
	N = float(len(im) * len(im[0]))
	media = 1/N * A
	#covarianza = 1/N * (B - (1/N) * A**2)
	return media


#Programa principal ----------

im = misc.imread('Test/2.tif')	#se carga la imagen
h = histograma(im)		#se genera el array del histograma
#print "media de misc ", im.mean(), "minimo de misc: ", im.min(), "maximo de misc: ", im.max()
#minimo = min_histograma(h)
copy = np.copy(im)		#se crea una copia de la imagen para poder comparar resultados 
media = umbral_fondo(h)
del_fondo(copy, media)
contraste(copy, copy.min(), media)
h1 = histograma(copy)
#misc.imsave('Test/Resultados/'+str(i)+'.tif',copy)




x = np.arange(0,256,1)	#se generan los valores 'x' de la grafica 
#x1 = np.arange(0,int(media),1)	#se generan los valores 'x' de la grafica 
pl.subplot(312)
pl.plot(x,h)			#se grafica
pl.grid()
pl.subplot(313)
pl.plot(x[:254],h1[:254])
pl.grid()				#anexamos un grid a la grafica
pl.subplot(321)
pl.axis('off')
pl.title('Original')
pl.imshow(im)
pl.subplot(322)
pl.axis('off')
pl.title('Resultado')
pl.imshow(copy)
pl.show()



#Aplicar suavizado inicial
#recorte de contraste dinamico 2%-3%