import math 
import json
import random
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

teach_amount = 50


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
        self.iter = 0
        self.it = []
        self.val = []

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
        inputs = inpt[:]
        inputs.append(1)       
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + self.k*(wish-res)*inputs[i]

if __name__ == '__main__':  
    print ('---------new init-----------')
    
    data = None

    with open('./../sample/sample.json') as sample:
        data = json.load(sample)
        sample.close

    classes = data['classes']
    nodes = []
    zeros = []
    for i in range(data['dimenision']+1): zeros.append(0)
    values = []

    for c in classes:
        values.append(data[c])
        nodes.append(Node(zeros[:], func, 0.02, c)) #0.05 -- скорость обучения

    epochs = []
    errors = []

    epoch = 0
    last = 1

    mem = []
    min_number = 1

    for node in nodes: mem.append([])

    while(True):
        epoch +=1
        print ('epoch: {}'.format(epoch))
        
        #эпоха
        for i in range(teach_amount):
            for c in range(len(values)):
                # print('class {}'.format(c))
                point = values[c][i]
                for cc in range(len(values)):
                    if cc!=c: 
                        nodes[cc].teach(point,0)
                        # print ('{}: {}'.format(nodes[cc].className,nodes[cc].w))
                    else:
                        nodes[c].teach(point,1)
                        # print ('{}: {}'.format(nodes[cc].className,nodes[cc].w))

        #проверка
        err_num = 0
        for c in range (len(values)):
            for point in values[c][teach_amount:]:
                res = []
                for node in nodes:
                    res.append(node.do(point))
                answer = res.index(max(res))
                if answer!= c: err_num+=1

        epochs.append(epoch)
        errors.append(err_num)
        plt.clf()
        plt.plot(epochs,errors,'b')
        plt.draw()
        plt.pause(0.1)

        #meta

        points_number = 0
        for c in range (len(values)): points_number+=len(values[c])
        
        if err_num / points_number < min_number:
            print ('!!!!!')
            min_number = err_num / points_number
            for index in range(len(nodes)):
                mem[index]=nodes[index].w[:]
                print (nodes[index].w)

        print ('errors: {}'.format(err_num))
        last = err_num

        #exit
        if err_num / points_number < 0.05: break 
        if last < err_num and epoch > 30: break
        if epoch > 50: break

    #final

    for index in range(len(nodes)):
        nodes[index].w=mem[index][:]
        print(nodes[index].w)

    err_num = 0
    for c in range (len(values)):
        print ('this is {}'.format(c))
        for point in values[c]:
            res = []
            for node in nodes:
                res.append(node.do(point))
            answer = res.index(max(res))
            print(answer)
            if answer!= c: err_num+=1

    print (err_num)
    plt.draw()
    plt.pause(0)