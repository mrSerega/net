import random as rnd
from ast import literal_eval
import decimal
import math
import numpy as np
from itertools import groupby
import struct

def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64
    return struct.unpack('>d', bf)[0]

def int_to_bytes(n, minlen=0):  # helper function
    """ Int/long to byte string. """
    nbits = n.bit_length() + (1 if n < 0 else 0)  # plus one for any sign bit
    nbytes = (nbits+7) // 8  # number of whole bytes
    b = bytearray()
    for _ in range(nbytes):
        b.append(n & 0xff)
        n >>= 8
    if minlen and len(b) < minlen:  # zero pad?
        b.extend([0] * (minlen-len(b)))
    return bytearray(reversed(b))  # high bytes first

# tests

def float_to_bin(f):
    """ Convert a float into a binary string. """
    ba = struct.pack('>d', f)
    # ba = bytearray(ba)  # convert string to bytearray - not needed in py3
    s = ''.join('{:08b}'.format(b) for b in ba)
    return s
    # return s[:-1].lstrip('0') + s[0] # strip all leading zeros except for last

DEUS = [-11,0]
COUNT = 10
STEP = 0.1

# for f in 0.0, 1.0, -14.0, 12.546, 3.141593:
#     binary = float_to_bin(f)
#     print('float_to_bin(%f): %r' % (f, binary))
#     float = bin_to_float(binary)
#     print('bin_to_float(%r): %f' % (binary, float))
#     print('')

#config me here ----------------------------------------------------
mods = ['booth','rosenbrock','sphere', 'rastrigin','eggholder', 'ackley' , 'bukin']

mode = 'bukin'

MUTATE_MODE = 0
CROSS_TYPE = 1
pop = 10    #population amount                                         
gen_len = 64      #(do not config)
p = 0.05           #mutation 

if mode == 'booth':
    d = 2                                #demenision of var    (do not config)                                      
    ranges = [[-10, 10], [-10, 10]]      #(do not config)
    def f(v):
        return float((decimal.Decimal(v[0])+decimal.Decimal(2*v[1])-7)**2 + (decimal.Decimal(2*v[0])+decimal.Decimal(v[1])-5)**2)       #ranges for each dem
elif mode == 'rosenbrock':
    d = 3                                      #demenision of var  
    ranges = [[-10,10] for el in range(d)] 
    def f(v):
        s = 0
        vv = [decimal.Decimal(el) for el in v]
        for i in range(d-1):
            first = (vv[i+1] - vv[i]**2)**2
            second = (1 - vv[i])**2
            s += (100*first + second)
            # s += float(100 * (decimal.Decimal(v[i+1])-decimal.Decimal(v[i])**2)**2 + decimal.Decimal((1-v[i]))**2)
        return float(s)
elif mode == 'sphere':
    d = 3
    ranges = [[-10,10] for el in range(d)] 
    def f(v):
        s = 0
        for i in range(d):
            s+=float(decimal.Decimal(v[i])**2)
        return s
elif mode == 'rastrigin':
    d = 4
    ranges = [[-5.12,5.12] for el in range(d)] #do not config
    def f(v):
        try:
            s = decimal.Decimal(0)
            for i in range(d):
                arg = float(2 * 3.14 * float(v[i]))
                # print('v: {}'.format(v[i]))
                # print('arg: {}'.format(arg))
                coss = math.cos(arg)
                fir = decimal.Decimal(v[i])**2
                s += (fir - 10 * decimal.Decimal(coss))
            s +=10 * d
            return float(s)
        except:
            return float('Inf')
elif mode == 'eggholder':
    d = 2
    ranges = [[-512,512],[-512,512]]
    def f(v):
        v147 = v[1]+47
        first = -(v147) * math.sin(math.sqrt(abs(v[0]/2 + v147)))
        second = v[0]*math.sin(math.sqrt(abs(v[0]-v147)))
        return  first - second
elif mode == 'ackley':
    d = 2
    ranges = [[-5,5],[-5,5]]
    def f(v):
        try:
            vv = [decimal.Decimal(el) for el in v]
            firstxx = vv[0]**2
            firstyy = vv[1]**2
            sq = math.sqrt(decimal.Decimal(0.5)*(firstxx+firstyy))
            first = -20*math.exp(-0.2 * sq)
            cosargx = 2*math.pi*float(vv[0])
            secondcosx = math.cos(cosargx)
            cosargy = 2*math.pi*float(vv[1])
            secondcosy = math.cos(cosargy)
            second = math.exp(0.5 * (secondcosx + secondcosy))
            third = math.e + 20
            return  first - second + third
        except Exception:
            return float('inf')
elif mode == 'bukin':
    MUTATE_MODE = 1
    pop = 100
    d = 2
    ranges = [[-15,5],[-3,3]]
    def f(v):
        x = float(v[0])
        y = float(v[1])
        xx = float(decimal.Decimal(x)**2)
        l = y - 0.01 * xx
        left = np.sqrt( abs( l ) )
        right = 0.01 * abs( x + 10 )
        return 100 * left + right


#config me here ----------------------------------------------------

def init_sample(d, ranges, amount):
    return [[rnd.uniform(ranges[index][0], ranges[index][1]) for index in range(d)] for creacher in range(amount)]



def cross(mom, dad, cross_type=0):
    if cross_type == 0:
        bin_mom = ''.join([float_to_bin(gen) for gen in mom])
        bin_dad = ''.join([float_to_bin(gen) for gen in dad])
        l = len(bin_mom)
        k = rnd.randint(0,l)
        first = bin_mom[0:k]+bin_dad[k+1:l]
        second = bin_dad[0:k]+bin_dad[k+1:l]
        gen_nums = int(l / gen_len)
        f = []
        s = []
        for index in range(gen_nums):
            f.append(bin_to_float(first[index*gen_len:(index+1)*gen_len]))
            s.append(bin_to_float(second[index*gen_len:(index+1)*gen_len]))
        # print ('----')
        # print (mom)
        # print (dad)
        # print (s)
        # print (f)
        return s, f   
    elif cross_type == 1:
        bin_mom = []
        for gen in mom: bin_mom.append(float_to_bin(gen))
        bin_dad = []
        for gen in dad: bin_dad.append(float_to_bin(gen))

        res = [[],[]]

        for gen_index in range(len(bin_mom)):
            l = len(bin_mom)
            k = rnd.randint(0,l)
            first = bin_mom[gen_index][:k]+bin_dad[gen_index][k:]
            second = bin_dad[gen_index][:k]+bin_dad[gen_index][k:]
            res[0].append(bin_to_float(first))
            res[1].append(bin_to_float(second))
        
        return res[0], res[1]
    elif cross_type == 2:
        child1 = [mom[1],dad[0]]
        child2 = [dad[1],mom[0]]
        deusex = rnd.random()
        if deusex < 0.0001:
            DEUS[0] += STEP
            DEUS[1] += STEP
            return DEUS, DEUS
        return child1, child2

if mode == 'bukin':
    CROSS_TYPE = 2
    ranges = [[-15,5],[-3,3]]

def make_love(population):
    rnd.shuffle(population)
    l = len(population)
    for index in range(int(l/2)):
        siblings = cross(population[index],population[l-1-index], cross_type = CROSS_TYPE)
        population.append(siblings[0])
        population.append(siblings[1])

def select(population):
    population.sort(key=f)
    for dot_index in range(len(population)):
        for el_index in range(len(population[dot_index])):
            if (population[dot_index][el_index] < ranges[el_index][0]):
                population[dot_index][el_index] = ranges[el_index][0]
            elif (population[dot_index][el_index] > ranges[el_index][1]):
                population[dot_index][el_index] = ranges[el_index][1]

    new_pop = [el for el, _ in groupby(population)]
    population = new_pop
    if len(population) < pop: return population
    return population[0:pop]

def mutate(creacher, mutate_mode = 1):
    if mutate_mode == 0:
        bin_cre = ''.join([float_to_bin(gen) for gen in creacher])
        for index in range(len(bin_cre)):
            isMutate = rnd.random()
            # print (isMutate)
            if 1-isMutate > p:
                bin_cre = list(bin_cre)
                if bin_cre[index] == '0': bin_cre[index] = '1' 
                if bin_cre[index] == '1': bin_cre[index] = '0' 
                bin_cre = ''.join(bin_cre)
        l = len(bin_cre)
        gen_nums = int(l / gen_len)
        c = []
        for index in range(gen_nums):
            c.append(bin_to_float(bin_cre[index*gen_len:(index+1)*gen_len]))
        # print (c)
        return c
    elif mutate_mode == 1:
        new_creacher = creacher[:]
        for index in range(len(new_creacher)):
            new_creacher[index] += rnd.random()* -0.5 * rnd.random()*10
        return new_creacher

def opt_creacher(dot):
    d = 0.0001    #   Дельта для поиска производной
    step = 2

    antiGrad = []

    ans = dot[:]
    
    for index in range(len(dot)):
        tmp_dot_left = dot[:]
        tmp_dot_right = dot[:]
        tmp_dot_right[index] += d
        tmp_dot_left[index] -= d
        grad = f(tmp_dot_right) - f(tmp_dot_left)
        grad /= 2.0
        antiGrad.append(-grad)

    for index in range(len(dot)):
        ans[index] += step * antiGrad[index]
    return ans


def opt_pop(sample):
    l = len(sample)
    for index in range(l):
        sample.append(opt_creacher(sample[index]))

if __name__ == '__main__':
    # print (f([0.1,0.1,0.1]))
    sample = init_sample(d, ranges, pop)
    # sample = select(sample) 
    # print (sample)
    # print (f([-9.966092763122655, 0.9933380279085775]))
    # input()
    while (True):
        make_love(sample)
        length = len(sample)
        for index in range(length): sample.append(mutate(sample[index][:], mutate_mode = MUTATE_MODE))
        opt_pop(sample)
        sample = select(sample)        
        # output
        val = "{0:.3f}".format(f(sample[0]))
        bad = "{0:.3f}".format(f(sample[-1]))
        dot = ["{0:.6f}".format(el) for el in sample[0]]
        print ('values: {}|{} / {}'.format(val,bad,dot))
    

