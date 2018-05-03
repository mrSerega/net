import numpy as np
import random
import json
import matplotlib.pyplot as plt
import copy
import sys
from sklearn.datasets import load_iris

debug = False

def loadData(path_to_input):
    pass

def normalizeData(X):
    tmparr = []
    tmparr = np.array(tmparr)
    for i in range (len(X)):
        tmparr=np.append(tmparr,X[i])
    # print(tmparr.len())
    mean = np.mean(tmparr)
    std = np.std(tmparr)
    for i in range (len(X)):
        X[i]=(X[i] - mean) / std
    return X

def encode(answer, class_number):
    ans = [0 for index in range(class_number)]
    ans[answer] = 1
    return ans

def sigm(x):
    return 1 / (1+np.exp(-x))

def dsigm(x):
    return sigm(x) * (1-sigm(x))

def getAnswer(answer):                           #[a1, a2]
    max_value = max(answer)
    return [1 if max_value == el else 0 for el in answer]

def loss(out, true_out):
    # print (out)
    # print (true_out)
    eps = 1e-10
    size = len(out)
    out = np.clip(out, eps, 1-eps)
    s = [true_out[index] * np.log(out[index])+(1-true_out[index]) * (np.log(1-out[index])) for index in range(size) ]
    s = -1 * sum(s) / size
    # print (s)
    return s

class Layer:

    def __init__(self,
        w,              #weight vector          [[w01, w12, w02], [w10, w11, w12]]
        b,              #cont input vector      [[b0], [b1]]
        ):
        self.W = np.array(w, dtype = float)
        self.B = np.array(b, dtype = float)

    def doForward(self,
        x               #input vector           [[x0],[x1],[x2]]
        ):
        self.X = np.array(x,dtype = float)
        self.Z = self.W.dot(self.X) + self.B
        # self.A = np.array([[sigm(el)] for el in self.Z])
        self.A = sigm(self.Z)
        # print ('A {}'.format(self.A))
        return self.A

    def doBackward(self,
        grad            #grad from next layer
        ):              

        # print ('self.Z '.format(self.Z))
        # print ('***')
        # print ('grad: {}'.format(grad))      
        # print ('***')
        dadz = np.array([dsigm(self.Z) * grad]).transpose()
        # print('dadz: {}'.format(dadz))
        

        # if debug: print('dw:\n{}'.format(self.dw))
        self.db = dadz.sum()                               ### 
        # if debug: print('db:\n{}'.format(self.db))
        # print ('dadz {}'.format(dadz))

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(self.X.shape[0], 1)
        else:
            self.X = np.transpose(self.X)

        # print ('X {}'.format(self.X))
        self.dw = dadz.dot(self.X.transpose())
        # print ('***')
        # print (self.dw)
        # print ('dick: {}'.format(np.dot(np.transpose(self.W),dadz).transpose()))
        # if debug: print('dw:\n{}'.format(self.dw))
        # print ('***')
        return np.dot(np.transpose(self.W),dadz).transpose()[0]
        

        # if debug: print('back grad: {}'.format(grad))
        # if debug: print('trans: {}'.format(w.transpose()))

        # if debug: print('back ans: {}'.format(grad.dot(dadz)))

        # return np.array(grad.dot(dadz)) #dot(w.transpose()))
        
class Net:
    def __init__(self,
        layers,         # [num of neurons] (first - num of inputs)
        speed           # learning rate
        ):
        self.loss_g = []
        self.layers = []
        self.speed = speed
        for l in range(1,len(layers)):
            lines = []                      #make W
            for cur in range(layers[l]):
                line = []
                for prev in range(layers[l-1]):
                    line.append(random.random())
                lines.append(line)
            b = []
            for index in range(layers[l]):
                b.append(random.random()) #////////////////
            self.layers.append(Layer(lines,b))

    def forward(self,
        sample
        ):
        for l in self.layers:
            sample = copy.deepcopy(l.doForward(sample))
        return sample
    
    def backward(self,
        grad
        ):
        for l in range(len(self.layers)):
            # print ('###')
            # print (grad)
            # print ('###')
            grad = copy.deepcopy(self.layers[len(self.layers)-1-l].doBackward(grad))
        # print ('###')
        # print (grad)
        # print ('###')

    def update(self):
        for l in self.layers:
            # print ('----')
            # print('W\n{}'.format(l.W))
            # print('dw\n{}'.format(l.dw))
            l.W = l.W - l.dw * self.speed
            # print('W\n{}'.format(l.W))
            l.B = l.B - l.db * self.speed
            # print (l.db)
            # print ('----')

    def predict(self,
        input
        ):
        return self.forward(input)

    def teach(self,
        input,
        output
        ):
        eps = 1e-5
        ans = getAnswer(self.predict(input))
        true_ans = np.array(encode(output,len(ans)))
        # true_ans = np.array([[el] for el in true_ans])
        A1 = np.array(self.layers[-1].A, dtype=float)
        self.loss_g.append(loss(A1,true_ans))
        
        # print (loss(getAnswer(A1), encode(output,len(A1))))
        A1 = np.clip(A1, eps, 1-eps)
        da1 = - true_ans / A1 + (1-true_ans) / (1-A1)
        # print ('teach')
        # print (da1)
        # print ('teach')
        self.backward(da1)
        self.update()


if __name__ == '__main__':

    # iris = load_iris()
    # print (iris)

    
    #config me here

    train_part = 1
    show = False
    
    #^^^^^^^^^^^^^^

    mode_sample = 'i'

    if mode_sample == 'i':
        sample_name = 'iris.json'
    if mode_sample == 's':
        sample_name = 'sample.json'

    with open('./../sample (done)/{}'.format(sample_name)) as sample:
        data = json.load(sample)
        sample.close

    classes_names = data['classes']
    input_size = data['dimenision']
    output_size = len(classes_names)
    classes = []
    for c in classes_names: classes.append(data[c])
    classes = normalizeData(classes)

    if show:
        plt.figure(3)
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        for c in classes:
            plt.plot([el[0] for el in c],[el[1] for el in c],'o')

    sample = {
        'class_names': [],
        'class_index': [],
        'train': [],
        'check': []
    }
    sample_dots = []
    for index in range(len(classes_names)):
        sample['class_names'].append(classes_names[index])
        sample['class_index'].append(index)
        for jndex in range(len(classes[index])):
            sample_dots.append((classes[index][jndex],index))
    train_index = int(len(sample_dots) * train_part)
    random.shuffle(sample_dots)
    sample['train'] = sample_dots[:train_index]
    sample['check'] = sample_dots[train_index+1:]


    net = Net([input_size,5,output_size], 0.05) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    best_epoch = 0
    best_epoch_raiting = 1
    best_net = copy.deepcopy(net)
    mem = []
    raitnig = [[],[]]

    for epoch in range(1000):

        # random.shuffle(sample['train'])
        
        for dot in sample['train']: net.teach(dot[0],dot[1])
        
        over_all = len(sample['train'])
        miss_number = 0

        for dot in sample['train']:
            if not (getAnswer(net.predict(dot[0])) == encode(dot[1],len(sample['class_names']))):
                miss_number += 1
        
        for dot in sample['check']:
            if not (getAnswer(net.predict(dot[0])) == encode(dot[1],len(sample['class_names']))):
                miss_number += 1

        current_raiting = miss_number/over_all
        mem.append(current_raiting)

        if current_raiting < best_epoch_raiting:
            best_epoch = epoch
            best_net = copy.deepcopy(net)

        if epoch > 200: break

        if sample_name != 'iris.json':
            if epoch > 20:
                if abs(current_raiting-mem[epoch - 2]) < 0.005:
                    if abs(current_raiting-mem[epoch - 3]) < 0.005:
                        if abs(current_raiting-mem[epoch - 4]) < 0.005:
                            if abs(current_raiting-mem[epoch - 5]) < 0.005:
                                if abs(current_raiting-mem[epoch - 6]) < 0.005:
                                    # print (111)
                                    break
        print ('epoch {}: {}'.format(epoch, current_raiting))
        raitnig[0].append(epoch)
        raitnig[1].append(current_raiting)

    # print (raitnig)
    plt.figure(0)
    # plt.plot([index for index in range(len(net.loss_g))],[l for l in net.loss_g])
    # plt.figure(1)
    plt.plot([el for el in raitnig[0]],[el for el in raitnig[1]])
    
    colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks', 'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',]
    
    if show:
        plt.figure(2)
        step = 0.05
        amount = int(4 / step)
        for x in range(amount):
            for y in range(amount):
                plt.plot(   -2 + x*step,
                            -2 + y*step,
                            colors[getAnswer((best_net.predict([-2 + x*step,-2 + y*step]))).index(1)])
            sys.stdout.write('{}/{}\r'.format(x,amount))

    plt.show()
