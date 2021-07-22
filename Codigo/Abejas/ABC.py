import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.signal import argrelextrema
import matplotlib.patches as patches


class Nodo():
	'''Esta clase genera nodos aleatorios y en base a estos encuentra la mejor ruta 
	en base al algortimo Bee swarm'''

	'''Para su constructor se necesita conocer la meta y la posicion en la que se 
	partira, así mismo se debe seleccionar un numero de nodos, por default
	el programa tendra 25 nodos.'''
	def __init__(self, meta, pos_inicial, mpx1, mpx2, mpy1, mpy2, num_nodos=25):

		self.num_nodos = num_nodos
		self.meta = meta
		self.pos_inicial = pos_inicial
		self.pos = np.array([pos_inicial])

		self.sizex = abs(mpx2-1)
		self.sizey = abs(mpy2-1)

		
		'''Se definen las variables que se utilizaran globalmente, las cuales
		son el arreglo ruta, en donde se guardaran los nodos seleccionados.
		Mientras que la matriz nodo es la que contendra los nodos generados
		aleatoriamente, por ultimo la matriz muertos obtiene los nodos
		desechados. '''
		self.muertos = np.zeros([1,2])
		self.ruta = np.array([[self.pos[0][0],self.pos[0][1]]])
		self.nodo = np.zeros([self.num_nodos,2])

	def Restricciones(self,valorx,valory,iter):
		cont = 0
		# La primera restriccion es que la posicion y el nodo no 
		# deben compartir el mismo espacio.

		if ((self.pos_inicial[0]==valorx)&(self.pos_inicial[1]==valory)):
			cont += 1

		# La segunda restriccion es que la meta y el nodo no 
		# deben compartir el mismo espacio.
		if ((self.meta[0]==valorx)&(self.meta[1]==valory)):
			cont += 1

		# La tercera restriccion es que los nodos no deben 
		# repetirse.
		Maux = np.copy(self.nodo[:iter])
		for i in range(iter):
			if ((Maux[i][0]==valorx)&(Maux[i][1]==valory)):
				cont += 1

		# La cuarta restriccion es que los nodos no deben existir
		# en los obstaculos

		''' Se necesita contar con conocimiento del mapa para este caso 
		se usaran los obstaculos puntuale del archivo obstacles.npy'''

		obs = np.load('obstacles.npy')

		for i in range(obs.shape[0]-1):
			if ((obs[i][0]==valorx)&(obs[i][1]==valory)):
				cont += 1

		if cont>0:
			return False
		else:
			return True


	def Puntos(self):
		
		# Se generan puntos en todo el mapa con la funcion choice se obtienen
		# aleatoriamente y de forma discreta.
		for i in range(self.num_nodos):
			res = False
			while (res==False):
				x = np.random.choice(self.sizex)
				y = np.random.choice(self.sizey)
				res = self.Restricciones(x,y,i)
			self.nodo[i][0] = x
			self.nodo[i][1] = y
			


	def Grafica_NPM0(self):

		# Este grafico es unicamente para ver la posicion de los nodos
		# Figura
		fig, ax = plt.subplots()
	
		# Meta y posicion, estos valores son fijos
		plt.plot(self.meta[0],self.meta[1], marker="o", color="b",label = "Meta")
		plt.plot(self.pos_inicial[0],self.pos_inicial[1], marker="o", color="g",label = "Posición")
		# Grafica nodos
		for i in range(int(np.size(self.nodo)/2)):
			if i==0:
				plt.plot(self.nodo[i][0],self.nodo[i][1], marker="o", color="red",label = "Nodos")
			else:
				plt.plot(self.nodo[i][0],self.nodo[i][1], marker="o", color="red")
		# Grafica de obstaculos
		obs = np.load('obstacles.npy')
	
		for i in range(obs.shape[0]-1):
			if i==0:
				plt.plot(obs[i][0],obs[i][1], marker="o", color="k",label = "Obstaculos")
			else:
				plt.plot(obs[i][0],obs[i][1], marker="o", color="k")
		# Parametros de la grafica
		plt.title("Grafico de NODOS")
		plt.xlabel("Eje X")   # Inserta el título del eje X
		plt.ylabel("Eje Y")   # Inserta el título del eje Y
		plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
		plt.tight_layout()




	def Grafica_NPM2(self,num_r = 1):
		# Esta funcion retorna de manera grafica la ruta obtenida
		# Figura
		fig, ax = plt.subplots()
		
		# Grafica ruta
		for i in range(self.ruta.shape[0]-1):
			if i==0:
				plt.plot([self.ruta[i][0],self.ruta[i+1][0]],[self.ruta[i][1],self.ruta[i+1][1]], marker="o", color="y",label= "Ruta")
			else:
				plt.plot([self.ruta[i][0],self.ruta[i+1][0]],[self.ruta[i][1],self.ruta[i+1][1]], marker="o", color="y")
		plt.plot(self.meta[0],self.meta[1], marker="o", color="b",label = "Meta")
		plt.plot(self.pos_inicial[0],self.pos_inicial[1], marker="o", color="g",label = "Posición")

		# Grafica obstaculos
		obs = np.load('obstacles.npy')
		for i in range(obs.shape[0]-1):
			if i==0:
				plt.plot(obs[i][0],obs[i][1], marker="o", color="k",label = "Obstaculos")
			else:
				plt.plot(obs[i][0],obs[i][1], marker="o", color="k")
			
		# Parametros de la grafica
		plt.title("Grafico de la Ruta"+str(num_r))
		plt.xlabel("Eje X")   # Inserta el título del eje X
		plt.ylabel("Eje Y")   # Inserta el título del eje Y
		plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
		plt.tight_layout()

	def Funcion_Distancia(self):
		contw = 0
		cont = 0
		# Esta funcion encuentra la mejor ruta analizando cada nodo
		while (1):
			cont+=1
			PopCost = np.zeros(self.nodo.shape[0]+1,dtype=np.float64)
			PopCostM = np.zeros(3)
			for i in range(self.nodo.shape[0]):
				PopCost[i] = distance.euclidean(self.pos, self.nodo[i])
			PopCost[self.nodo.shape[0]] = distance.euclidean(self.pos, self.meta)
			
			ordenados = np.array(sorted(PopCost),dtype=np.float64)
			arg_orden = np.zeros(3,dtype=np.uint32)
			nodo_m=np.zeros([3,2])
			i=0
			for i in range(3):
				aux_arg = np.where(PopCost==ordenados[i])[0]
				if aux_arg.shape[0]>1:
					aux_arg = aux_arg[0]
				arg_orden[i] = int(aux_arg)
				if arg_orden[i] == PopCost.shape[0]-1:
					aux = np.array([[self.meta[0],self.meta[1]]])
					self.ruta = np.append(self.ruta,aux,axis=0)
					contw+=1
					break
				aux = np.array([[self.nodo[int(arg_orden[i])][0],self.nodo[int(arg_orden[i])][1]]])
				nodo_m[i]=self.nodo[int(arg_orden[i])]
				self.muertos = np.append(self.muertos,aux,axis=0)
			if contw>0:
				break
			for i in range(3):
				PopCostM[i] = distance.euclidean(self.meta, nodo_m[i])

			ordenados = np.array(sorted(PopCostM))

			arg = np.where(PopCostM==ordenados[0])[0]
			if arg.shape[0]>1:
					arg = arg[0]
			aux = np.array([[nodo_m[int(arg)][0],nodo_m[int(arg)][1]]])

			
			self.ruta = np.append(self.ruta,aux,axis=0)
			self.pos = aux
			self.nodo = np.delete(self.nodo, [int(arg_orden[0]),int(arg_orden[1]),int(arg_orden[2])],axis=0)

		return self.ruta
		
	def distancia_min(self):
		dis = 0
		for i in range(self.ruta.shape[0]-1):
			dis += distance.euclidean(self.ruta[i], self.ruta[i+1])
		return dis

	def Rango(self,num,min,max):
		if ((num>=min)&(num<=max)):
			return False
		else:
			return True



''' Para ejecutar el codigo defina el numero de nodos, rutas, meta y posicion'''
n_nodos = 25
num_rutas = 100
meta = np.array([17,18])
posicion = np.array([1,2])
''' Para obtener el mejor resultado se deben plantear diversas rutas, para eso 
cree una lista de objetos'''
nodo = []

'''Para saber cual es la mejor ruta, primero asigne un valor muy grande a una variable
se recomienda colocar infinito'''
bestruta = np.inf
aux_indice = 0

'''Cree un for con el numero de rutas deseado en este defina los objetos y sus metodos'''

for i in range(num_rutas):
	nodo += [i + 1] # incrementa el tamaño de la lista
	nodo[i]=Nodo(meta,posicion,0,20,0,20,num_nodos=n_nodos)
	nodo[i].Puntos() # Obtenemos los nodos 
	# nodo[i].Grafica_NPM0() # Se grafican los nodos
	auxr = nodo[i].Funcion_Distancia() # Se calcula la ruta
	# nodo[i].Grafica_NPM2(i+1) # Se grafica la ruta
	if bestruta > nodo[i].distancia_min(): # Se guarda la ruta mas baja 
		bestruta = nodo[i].distancia_min()
		aux_indice = i
		ruta = auxr

''' Ahora solo imprima los valores encontrados '''

print("La mejor ruta es la:",aux_indice+1,"Con una distancia de:",bestruta,"metros")
nodo[aux_indice].Grafica_NPM2(aux_indice+1)
print(ruta)
plt.show()
