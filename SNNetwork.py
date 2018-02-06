import numpy as np
import math
from Link import *
from AsyncSN import *


#class to construct the spiking neural network given an array whose length sets the number of layers, 
#and the values are the number of neurons on each layer, without considering the input layer
class SNNetwork:
	def __init__(self, netLayout, inputNeurons, terminals, inhibN, threshold, tau, timeStep,dinc):
		layersNumber = netLayout.shape

		#numberInhibN = inhibN

		#will have a size set by layersNumber[0]
		self.layers = list()
		for lyer in range(layersNumber[0]):
			neurons = netLayout[lyer]
			self.layers.append(np.empty((neurons),dtype=object))

			if lyer == 0:
				connections = inputNeurons
			else:
				connections = netLayout[lyer-1]

			for n in range(neurons):
				if lyer != layersNumber[0]:
					self.layers[lyer][n] = AsyncSN(connections, terminals, inhibN,threshold, tau,timeStep,dinc)
					inhibN -= 1
				else:
					self.layers[lyer][n] = AsyncSN(connections, terminals, 0)

	#returns the last firing times of a layer of neurons
	@classmethod
	def getFireTimesLayer(self, layer):
		noNeurons = layer.shape
		preSNFTime = np.zeros(noNeurons[0])
		#print('target:', t)
		for n in range(noNeurons[0]):
			#get the last element of the list storing the firing times of the neuron
			preSNFTime[n] = layer[n].getLastFireTime()

		return preSNFTime

	@classmethod
	def getTypesLayer(self, layer):
		noNeurons = layer.shape
		preSNTypes = np.zeros(noNeurons[0])

		for n in range(noNeurons[0]):
			#get the last element of the list storing the firing times of the neuron
			preSNTypes[n] = layer[n].type

		return preSNTypes


	def resetSpikeTimeNet(self):
		layersNo = len(self.layers)

		for l in range(layersNo):
			noNeurons = self.layers[l].shape
			for n in range(noNeurons[0]):
				self.layers[l][n].resetSpikeTimes()

	def resetSpikeTimeLayer(self, layer):
		noNeurons = layer.shape
		for n in range(noNeurons[0]):
			layer[n].resetSpikeTimes()

	def displaySNN(self,outputNeurons):
		#print ('------------ Displaying the network properties ------------')
		print("************* Number of Neurons at Iutput Layer: 9 **************")
		print("Number of Neurons at Output Layer :",outputNeurons)
		layersNumber = len(self.layers)
		for l in range(layersNumber):
			neurons = self.layers[l].shape
			print ("Number of Neurons at:", l, 'th Hidden Layer ', neurons[0])
			if (l== 1):
			  print("*************** Properties of Output Layer Neurons:**************")
			for n in range(neurons[0]):
				print (n, 'th Neuron ',  ' has the following properties:')
				self.layers[l][n].displaySN()

