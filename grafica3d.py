



from mpl_toolkits.mplot3d import Axes3D         # Cargo Axes3D de mpl_toolkits.mplot3d
from scipy.misc import imread                   # Cargo imread de scipy.misc
import numpy as np                              # Cargo numpy como el aliaas np
import matplotlib.pyplot as plt                 # Cargo matplotlib.pyplot  en el alias sp
import csv

temperatura = []
# Leo una imagen y la almaceno en imagen_superficial
imagen_superficial = imread('Test2/2.tif')

#Cargamos archivo de la imagen
with open('Test2/2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for linea in reader:
        l = []
        for row in linea:
        	if (row != ' '):
        		l.append(float(row))
        temperatura.append(np.array(l))
print temperatura[0][1]

# Creo una figura
plt.figure()

# Muestro la imagen en pantalla
plt.imshow(imagen_superficial)

# Aado etiquetas
plt.title('Imagen que usaremos de superficie')
plt.xlabel(u'# de pixeles')
plt.ylabel(u'# de pixeles')

# Creo otra figura y la almaceno en figura_3d
figura_3d = plt.figure()

# Indicamos que vamos a representar en 3D
ax = figura_3d.gca(projection = '3d')

# Creamos los arrays dimensionales de la misma dimension que imagen_superficial
X = np.linspace(0, imagen_superficial.shape[0], imagen_superficial.shape[0])
Y = np.linspace(0, imagen_superficial.shape[1], imagen_superficial.shape[1])
print imagen_superficial.shape[0]
# Obtenemos las coordenadas a partir de los arrays creados
X, Y = np.meshgrid(X, Y)

# Defino la funcion que deseo representar
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)
print type(temperatura[0])
print len(temperatura[0]), len(temperatura[1])
T = np.array(temperatura, copy=True, imagen_superficial.shape[1])
print "temperatura ", T.shape[0], Z.shape[1]
print "array Z", Z.shape[0], Z.shape[1]
print Z

# Reescalamos de RGB a [0-1]
imagen_superficial = imagen_superficial.swapaxes(0, 1) / 255. 

# meshgrid orienta los ejes al reves luego hay que voltear
ax.plot_surface(X, Y, Z, facecolors = np.flipud(imagen_superficial))

# Fijamos la posicion inicial de la grafica
ax.view_init(45, -35)

# Aadimos etiquetas
plt.title(u'Imagen sobre una grafica 3D')
plt.xlabel('Eje x')
plt.ylabel('Eje y')
# Mostramos en pantalla
plt.show()