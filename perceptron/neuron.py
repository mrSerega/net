import math

class Neuron:

    def func(self, x):
        if x>10: return 1
        elif x<-10: return 0
        else: return 1.0 / (1+math.exp(-x))

    def dfunc(self, x):
        return math.exp(-x) / (1 + math.exp(-x))

    def __init__(self, defaultSpeed):
        self.inputsW = []
        self.inputs = []
        self.speed = defaultSpeed
        self.grad = 0
        self.value = 0

    def getValue(self):
        return self.value
    
    def getGrad(self):
        return self.grad

    def addOutput(self, neuron):
        neuron.inputs.append(self)
        neuron.inputsW.append(1)

    def forward(self):
        s = 0
        
        for index in range(len(self.inputs)):
            s += self.inputsW[index] * self.inputs[index].getValue()
        self.value = self.func(s)

    def reverse(self):
        self.grad = self.dfunc(self.grad)

        for index in range(len(self.inputs)):
            inputsW[index] += inputs[index].getValue() * self.grad * self.speed
            inputs[index].grad += inputsW[index] * self.grad

        self.grad = 0

class X:
    
    def func(self, x):
        if x>10: return 1
        elif x<-10: return 0
        else: return 1.0 / (1+math.exp(-x))

    def dfunc(self, x):
        return math.exp(-x) / (1 + math.exp(-x))

    def __init__(self, defaultSpeed):
        self.value = 0
        self.speed = defaultSpeed

    def addOutput(self, neuron):
        neuron.inputs.append(self)
        neuron.inputsW.append(1)

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

class Y:
    
    def func(self, x):
        if x>10: return 1
        elif x<-10: return 0
        else: return 1.0 / (1+math.exp(-x))

    def dfunc(self, x):
        return math.exp(-x) / (1 + math.exp(-x))

    def __init__(self, defaultSpeed):
        self.inputs = []
        self.inputsW = []
        self.value = 0
        self.answer = 0
        self.speed = defaultSpeed

    def setAnswer(self, value):
        self.answer = value

    def getValue(self):
        return self.value

    def forward(self):
        s = 0
        
        for index in range(len(self.inputs)):
            s += self.inputsW[index] * self.inputs[index].getValue()
        self.value = self.func(s)

    def reverse(self):
        s = 0

        for index in range(len(self.inputs)):
            s += self.inputsW[index] * self.inputs[index].getValue()

        self.grad = (self.answer - self.value) * self.dfunc(s)

        for index in range(len(self.inputs)):
            self.inputsW[index] += self.inputs[index].getValue() * self.grad * self.speed
            self.inputs[index].grad += self.inputsW[index] * self.grad

        self.grad = 0