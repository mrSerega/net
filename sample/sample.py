# class sample generator
# usage: python sample.py <log_name.json> <config_name.json>
# config example: 
#   {
#       "seed": -1,                 зерно для генератора ПСЧ 
#       "try_num": 10000,           максимально количетсво попыток для создания класса
#       "dimenision": 2,            количетсво критериев точек класса
#       "amount": 5,                количество классов
#       "radius_range": [20,30],    интервал допустимых радиусов классов
#       "points_range": [40,50],    интревал допустимого количетсва точек классов
#       "min_distanse": 61,         минимальное рассточние между центрами классов
#       "max_distanse": 200,        максимальное расстоняие между центрами классов
#       "max_length": 1000          макисимальная длина измерения поля
#   }

import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import sys
import time

class Sample:
    seed = None
    try_num = 0
    dimenision = 0
    amount = 0
    radius_range = [0,0]
    points_range = [0,0]
    max_intersection = 0
    min_distanse = 0
    max_length = 0
    max_distanse = 0
    distribution_func = 0
    zero = []
    classes = []
    colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks', 'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',]
    
    def __init__(self,
                try_num, #количество попыток создать класс
                dimenision, #количество критериев
                radius_range, #интервал допустимых радиусов классов
                min_distanse, #минимальное расстояние между центрами классов
                max_length, #максимальная длина/ширина поля
                max_distanse, #максимальное расстояние между центрами классов
                points_range, #интервал допустимого числа точек в классе
                seed): #зерно рандома
        self.try_num = try_num
        self.dimenision = dimenision
        self.radius_range = radius_range
        self.min_distanse = min_distanse
        self.max_length = max_length
        self.max_distanse = max_distanse
        self.points_range = points_range
        if (seed >=0 ):
            self.seed = seed
        else:
            self.seed = time.time()
        random.seed(self.seed)
        for i in range(self.dimenision):
            self.zero.append(0)
  
    def getDistance(self, vector1, vector2):
        distanse = 0
        for i in range(len(vector1)):
            distanse += (vector1[i]-vector2[i])**2
        distanse = math.sqrt(distanse)
        return distanse

    def getRandomVector(self, rnd_max, zero_point):
        vector = []
        
        for i in range(self.dimenision):
            vector.append(random.random()-0.5)
            
        length = 0
        for i in range(self.dimenision):
            length += vector[i]**2
            
        length = math.sqrt(length)
        
        random_length = random.random() * rnd_max
        
        for i in range(self.dimenision):
            vector[i] = vector[i]/length
            vector[i] = vector[i]*random_length
            vector[i] = vector[i] + zero_point[i]
            
        return vector
    
    def createClass(self):
        
        bad_class = True

        center = []
        
        for i in range(self.try_num):
            
            center = self.getRandomVector(self.max_length, self.zero)
            
            if len(self.classes) == 0:
                bad_class = False
                break
                
            good_class = True

            for j in range(len(self.classes)):
                if self.getDistance(center,self.classes[j][0]) < self.min_distanse:
                    good_class = False
                    break

            for j in range(len(self.classes)):
                if self.getDistance(center,self.classes[j][0]) > self.max_distanse:
                    good_class = False
                    break
                
            if(good_class):
                bad_class = False
                break
                
        if(bad_class):
            print ('cant create new class')
            return
        
        rnd_rad = random.random()
        rnd_rad *= (self.radius_range[1]-self.radius_range[0])
        rnd_rad += self.radius_range[0]
        
        amount = random.random()
        amount *= (self.points_range[1]-self.points_range[0])
        amount += self.points_range[0]
        amount = round(amount)
        
        new_class = []
        
        new_class.append(center)
        for i in range(amount):
            new_class.append(self.getRandomVector(rnd_rad, center))
        
        self.classes.append(new_class)

    def showClasses(self):
        if self.dimenision != 2 and self.dimenision != 3:
            print ('plot only for 2d or 3d!')
            return
        
        if self.dimenision == 2:
            for i in range(len(self.classes)):
                x = [el[0] for el in self.classes[i]]
                y = [el[1] for el in self.classes[i]]
                plt.plot(x,y,self.colors[i])
        
        if self.dimenision == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            for i in range(len(self.classes)):
                x = [el[0] for el in self.classes[i]]
                y = [el[1] for el in self.classes[i]]
                z = [el[2] for el in self.classes[i]]
                ax.scatter(x,y,z,self.colors[i])
            
        plt.show()
    
    def logClasses(self):
        data = {}
        data['dimenision'] = self.dimenision                #meta: размерность точек
        data['seed'] = self.seed                            #mata: зерно ГПСЧ
        data['classes'] = []                                #meat: перечень имен классов
        for i in range(len(self.classes)):
            data['class{}'.format(i)] = self.classes[i]     #добавляем класс в перечень
            data['classes'].append('class{}'.format(i))     #meta: добавляем имя класса в перечень
        with open(sys.argv[1], 'w') as output:
             json.dump(data, output)

if __name__ == '__main__':

    if (len(sys.argv)!=3): 
        print ('use sample.py <log_name.json> <config_name.json>')
        exit(1)

    data = 0
    
    with open(sys.argv[2]) as config:
        data = json.load(config)
        config.close()

    sample = Sample(try_num = data['try_num'],
                    dimenision = data['dimenision'],
                    radius_range = data['radius_range'],
                    points_range = data['points_range'],
                    min_distanse = data['min_distanse'],
                    max_distanse = data['max_distanse'],
                    max_length = data['max_length'],
                    seed = data['seed']) 
    for i in range(data['amount']):
        sample.createClass()
    sample.logClasses()
    sample.showClasses()