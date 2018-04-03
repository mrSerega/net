import math 
import json
import random

def func(x):
    if x>10: return 1
    elif x<-10: return 0
    else: return 1.0 / (1+math.exp(-x))

class Node:
    w = []
    func = None
    k = 1

    def __init__(self,
                w,      #веса
                func,   #функция активации
                k,
                className):     #скорость обучения
        self.w = w
        self.func = func
        self.k = k
        self.className = className

    def do(self, inpt):
        inputs = inpt[:]
        inputs.append(1)
        if len(inputs) != len(self.w):
            raise Exception('different number of inputs and arguments')
        
        sum = 0
        for i in range(len(self.w)):
            sum+=self.w[i]*inputs[i]

        return self.func(sum)

    def teach(self, inpt, wish):
        res = self.do(inpt)
        print ('{}: {}'.format(self.className,res))
        inputs = inpt[:]
        inputs.append(1)
        for i in range(len(self.w)):
            print (self.k*(wish-res)*inputs[i])
            self.w[i] = self.w[i] + self.k*(wish-res)*inputs[i]

if __name__ == '__main__':

    data = None

    with open('./../sample/sample.json') as sample:
        data = json.load(sample)
        sample.close

    classes = [data['class0'],data['class1']]

    node1 = Node([0,0,0], func, 0.005, 'first')
    node2 = Node([0,0,0], func, 0.005, 'second')

    print ('teaching...')

    for i in range (100):
        for c in range(len(classes)):
            point = random.choice(classes[c])
            if(c == 0):
                print ('class: {}'.format(c))
                node1.teach(point,1)
                node2.teach(point,0)
            else:
                print ('class: {}'.format(c))
                node1.teach(point,0)
                node2.teach(point,1)

    print ('********')
    print (node1.w)
    print (node2.w)
    print ('********')

    for c in range(len(classes)):
        print ('this is {}:'.format(c+1))
        for point in classes[c]:
            res1 = node1.do(point)
            res2 = node2.do(point)
            print('res1: {}, res2: {}'.format(res1, res2))
            if res1>res2: print (1)
            else: print (2)
            print ('---')