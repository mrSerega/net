import random
import math
import matplotlib.pyplot as plt

class Sample:
    
    try_num = 0;
    dimenision = 0
    amount = 0
    radius_range = [0,0]
    points_range = [0,0]
    max_intersection = 0
    min_distance = 0
    max_distance = 0
    distribution_func = 0
    zero = []
    classes = []
    colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks', 'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',]
    
    def __init__(self):
        self.try_num = 1000
        self.dimenision = 2
        self.amount = 2
        self.radius_range = [10,50]
        self.max_intersection = 10
        self.min_distance = 200
        self.max_distance = 1000
        self.distribution_func = 0
        self.points_range = [10,100]
        for i in range(self.dimenision):
            self.zero.append(0)
  
    def getDistance(self, vector1, vector2):
        distance = 0;
        for i in range(len(vector1)):
            distance += (vector1[i]-vector2[i])**2
        distance = math.sqrt(distance)
        return distance

    def getRandomVector(self, rnd_min, rnd_max, zero_point):
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
#            vector[i] = vector[i]+rnd_min
            vector[i] = vector[i] + zero_point[i]
            
        return vector
    
    def createClass(self):
        
        bad_class = True
        
        center = []
        
        for i in range(self.try_num):
            
            center = self.getRandomVector(self.min_distance, self.max_distance, self.zero)
            
            if len(self.classes) == 0:
                bad_class = False
                break
                
            good_class = True
                
            for j in range(len(self.classes)):
                if self.getDistance(center,self.classes[j][0]) < self.min_distance:
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
            new_class.append(self.getRandomVector(0, rnd_rad, center))
        
        self.classes.append(new_class)
        
    def showClasses(self):
        if self.dimenision != 2:
            print ('plot only for 2d!')
            return
        
        for i in range(len(self.classes)):
            x = [el[0] for el in self.classes[i]]
            y = [el[1] for el in self.classes[i]]
            plt.plot(x,y,self.colors[i])
            
        plt.show()
            
        
    
sample = Sample()
for i in range(20):
    sample.createClass()
sample.showClasses()