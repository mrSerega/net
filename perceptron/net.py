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

        for index in range(1,len(self.layers)):
            for neuron in self.layers[index]:
                neuron.forward()

        if getAnswer:
            return [neuron.getValue() for neuron in self.layers[-1]]

    def teach(self, inputs, outputs):
        self.predict(inputs, 0)

        for index in range(len(self.layers[-1])):
            self.layers[-1][index].setAnswer(outputs[index])

        for index in range(len(self.layers)-1,0,-1):
            # print ('layer: {}'.format(index))
            for neuron in self.layers[index]:
                neuron.reverse()
                # print ('W: {}'.format(neuron.inputsW))
            # print ('----------')
                
    
if __name__ == '__main__':
        #config me here
        layerNumber = 2
        inputNumbers = 2
        layerSize = 2
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

        print ('===')
        for neuron in net.layers[-1]:
            print(neuron.inputsW)
        print ('===')

        net.teach([1/2000,0], [0, 1])
        net.teach([-1/2000,0], [1, 0])
        net.teach([10/2000,0], [0, 1])
        net.teach([-10/2000,0], [1, 0])
        net.teach([100/2000,0], [0, 1])
        net.teach([-100/2000,0], [1, 0])
        net.teach([1000/2000,0], [0, 1])
        net.teach([-1000/2000,0], [1, 0])

        # for n in net.layers[-1]:
        #     print (n.inputsW)

        print ('prediction: {}'.format(net.predict([500/2000,0])))
        print ('prediction: {}'.format(net.predict([-500/2000,0])))