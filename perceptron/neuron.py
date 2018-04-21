import math

class Neuron:

    def func(self, x):
        if x>10: return 1
        elif x<-10: return 0
        else: return 1.0 / (1+math.exp(-x))

    def dfunc(self, x):
        return self.func(x) * (1-self.func(x))

    def __init__(self, defaultSpeed):
        self.b = 1
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
        s += self.b
        self.value = self.func(s)

    def reverse(self):
        # print ('grad: {}'.format(self.grad))
        s = 0
        
        for index in range(len(self.inputs)):
            s += self.inputsW[index] * self.inputs[index].getValue()
        s += self.b

        self.grad = self.dfunc(s) * self.grad

        for index in range(len(self.inputs)):
            self.inputsW[index] += self.inputs[index].getValue() * self.grad * self.speed
            self.inputs[index].grad += self.inputsW[index] * self.grad

        self.b += self.grad * self.speed

        self.grad = 0

class X:

    def __init__(self, defaultSpeed):
        self.value = 0
        self.speed = defaultSpeed
        self.grad = 0

    def addOutput(self, neuron):
        neuron.inputs.append(self)
        neuron.inputsW.append(1)

    def setValue(self, value):
        self.value = value
        # print ('V: {}'.format(self.value))

    def getValue(self):
        return self.value

class Y:
    
    def func(self, x):
        if x>10: return 1
        elif x<-10: return 0
        else: return 1.0 / (1+math.exp(-x))

    def dfunc(self, x):
        return self.func(x) * (1-self.func(x))

    def __init__(self, defaultSpeed):
        self.b = 1
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
        # print ('forward action  ****')
        for index in range(len(self.inputs)):
            s += self.inputsW[index] * self.inputs[index].getValue()
            # print ('value: {}'.format(self.inputs[index].getValue()))
        s += self.b

        self.value = self.func(s)

    def reverse(self):
        # print ('reversereversereverse')
        s = 0

        for index in range(len(self.inputs)):
            s += self.inputsW[index] * self.inputs[index].getValue()
        s += self.b

        # print ('s: {}'.format(s))
        # print ('ss: {}'.format(self.dfunc(s)))
        # print ('d: {}'.format(self.answer - self.value))

        # print ('func1: {}'.format(self.dfunc(s)))
        # print ('dfunc: {}'.format(self.func(s) * (1 - self.func(s))))

        self.grad = (self.answer - self.value)

        for index in range(len(self.inputs)):
            self.inputsW[index] += self.inputs[index].getValue() * self.grad * self.speed
            self.inputs[index].grad += self.inputsW[index] * self.grad
            # print ('getValue: {}'.format(self.inputs[index].getValue()))
            # print ('grad: {}'.format(self.grad))
            # print ('dw: {}'.format(self.inputs[index].getValue() * self.grad * self.speed))

        self.b += self.grad * self.speed

        self.grad = 0