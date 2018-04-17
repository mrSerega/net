import neuron as _

class Network:
    def __init__(self, layersAmount, defaultSpeed):
        self.layers = []
        self.speed = defaultSpeed
        for index in range(layersAmount):
            self.layers.append([])

    def addNeuron(self, layerNumber):
        if layerNumber == 0:
            self.layers[0].append(_.X(self.speed))
        elif layerNumber == len(self.layers)-1 or layerNumber == -1:
            self.layers[-1].append(_.Y(self.speed))
        elif layerNumber > 0 and layerNumber < len(self.layers)-1:
            self.layers[layerNumber].append(_.Neuron(self.speed))

    def doMash(self):
        for index in range(len(self.layers)-1):
            for currentLayerNeuron in self.layers[index]:
                for nextLayerNeuron in self.layers[index+1]:
                    currentLayerNeuron.addOutput(nextLayerNeuron)

    def predict(self, inputs, getAnswer = 1):
        for index in range(len(self.layers[0])):
            self.layers[0][index].setValue(inputs[index])

        for index in range(1,len(self.layers)-1):
            for neuron in self.layers[index]:
                neuron.forward()

        if getAnswer:
            return [neuron.getValue() for neuron in self.layers[-1]]

    def teach(self, inputs, outputs):
        self.predict(inputs, 0)
        
        for index in range(len(self.layers[-1])):
            self.layers[-1][index].setAnswer(outputs[index])

        for index in range(len(self.layers)-1,1,-1):
            for neuron in self.layers[index]:
                neuron.reverse()
    
if __name__ == '__main__':
        #config me here
        layerNumber = 3
        inputNumbers = 2
        layerSize = 3
        outputNumber = 2
        speed = 0.05
        
        net = Network(layerNumber, speed)

        for index in range(inputNumbers):
            net.addNeuron(0)

        for index in range(layerNumber-2):
            for jndex in range(layerSize):
                net.addNeuron(index+1)

        for index in range(outputNumber):
            net.addNeuron(-1)

        net.doMash()

        net.teach([1,1], [2,2])
        net.teach([1,1], [2,2])
        net.teach([1,1], [2,2])
        net.teach([1,1], [2,2])

        print (net.predict([1,1]))