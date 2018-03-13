import math 
import json

def func(x):
    return 1.0 / (1+math.exp(-x))

class Node:
    w = []
    func = None
    k = 1

    def __init__(self,
                w,      #веса
                func,   #функция активации
                k):     #скорость обучения
        self.w = w
        self.func = func
        self.k = k

    def do(self, inpt):
        inputs = inpt[:]
        inputs.append(1)
        if len(inputs) != len(self.w):
            raise Exception('different number of inputs and arguments')
        
        sum = 0
        for i in range(len(self.w)):
            sum+=self.w[i]*inputs[i]

        return self.func(sum)

    def teach(self, err):
        for i in range(len(self.w)):
            self.w[i] += self.k*err

def checkError(classification, trueClass, numberOfClasses):
    step = 1.0 / numberOfClasses
    trueClassRange = [step*trueClass, step*(trueClass+1)]
    print ('node: {}, true: {}, dif: {}'.format(classification,(trueClassRange[0]+trueClassRange[1]) / 2.0,  -(classification - (trueClassRange[0]+trueClassRange[1]) / 2.0)))
    return -(classification - (trueClassRange[0]+trueClassRange[1]) / 2.0)

def checkClass(classification, trueClass, numberOfClasses):
    step = 1.0 / numberOfClasses
    trueClassRange = [step*trueClass, step*(trueClass+1)]
    if classification > trueClassRange[0] and classification < trueClassRange[1]: return True
    return False

if __name__ == '__main__':

    data = None

    with open('./../sample/sample.json') as sample:
        data = json.load(sample)
        sample.close

    classes = [data['class0'],data['class1']]

    node1 = Node([0,0,0], func, 0.005)

    for point in classes[0]:
        node1.teach(checkError(node1.do(point),0,len(classes)))
        print (node1.w)

    node2 = Node([0,0,0], func, 0.005)

    for point in classes[1]:
        node2.teach(checkError(node2.do(point),1,len(classes)))
        print (node2.w)


    for c in range(len(classes)):
        for point in classes[c]:
            res1 = node1.do(point)
            res2 = node2.do(point)
            print ('res1: {}, res2: {}. c: {}'.format(res1,res2,c))
            if (abs(res1-0.25) < abs(res2-0.75)) and c == 0: print(True) 
            elif (abs(res1-0.25) >= abs(res2-0.75)) and c == 1: print(True)
            else:
                print(False) 
    