import numpy as np
import pandas
import pickle
from SNNetwork import *
from DataProc import *
from SpikeProp import *
import sys

def main():

    data, target = DataProc.readData('wbcFull.data', 9) # number of input layer neuron = 9
    #data, target = DataProc.readData('maligant.data', 9)
    #data, target = DataProc.readData('bening.data', 9)

    # add an extra column as the bias
    inputdata = DataProc.addBias(data)
    minValue = np.min(np.min(data, axis=1), axis=0)
    maxvalue = np.max(np.max(data, axis=1), axis=0)
    #print(minValue, maxvalue)

    deltaT = maxvalue - minValue
    fullSample = data.shape[0]
    sample=int(fullSample / 2)
    trainingInput = inputdata[:sample, :]
    trainingTarget = target[:sample, :]
    print(sample)
    testingInput = inputdata[sample:, :]
    testingTarget = target[sample:, :]
    timeStep = 1
    dinc = 2
    learningRate = 0.01
    epochs = 1
    hidNeuron = 8
    tau = 11
    threshold = 2
    terminals = 16

    benign = 7  # it will go 10- 38 range
    maligant =8
    inputNeurons = data.shape
    outputNeuron = 1
    netLayout = np.asarray([hidNeuron, outputNeuron])

    # set the number of inhibitory neurons to set in the network
    inhibN = 1
    SpikeProp(outputNeuron, timeStep,deltaT, tau, terminals)
    #SpikeProp.setTimeLimit(deltaT, tau, terminals)
    net = SNNetwork(netLayout, inputNeurons[1], terminals, inhibN,threshold, tau, timeStep,dinc)
    #net.displaySNN(outputNeuron)

    SpikeProp.train(net, trainingInput, trainingTarget, learningRate, epochs,sample, benign, maligant)
    # save the model to disk
    filename = 'finalNet.sav'
    pickle.dump(net, open(filename, 'wb'))
    print("*****************************Training Completed *********************************")
    loaded_model = pickle.load(open(filename, 'rb'))
    SpikeProp.test(loaded_model, testingInput, testingTarget, learningRate, sample, benign, maligant)


if __name__ == "__main__":
	main()