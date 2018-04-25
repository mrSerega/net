import random as rnd
from ast import literal_eval
import decimal
import math

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

# for f in 0.0, 1.0, -14.0, 12.546, 3.141593:
#     binary = float_to_bin(f)
#     print('float_to_bin(%f): %r' % (f, binary))
#     float = bin_to_float(binary)
#     print('bin_to_float(%r): %f' % (binary, float))
#     print('')

#config me here ----------------------------------------------------
mods = ['booth','rosenbrock','sphere', 'rastrigin','eggholder']

mode = 'rosenbrock'

pop = 2000     #population amount                                         
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

def make_love(population):
    rnd.shuffle(population)
    l = len(population)
    for index in range(int(l/2)):
        siblings = cross(population[index],population[l-1-index], cross_type = 1)
        population.append(siblings[0])
        population.append(siblings[1])

def select(population):
    sample.sort(key=f)
    # print (population)
    # for dot_index in range(len(population)):
    #     try:
    #         for el_index in range(len(population[dot_index])):
    #             try:
    #                 if (population[dot_index][el_index] < ranges[el_index][0]) or (population[dot_index][el_index] > ranges[el_index][1]):
    #                     population.pop(dot_index)
    #             except Exception:
    #                 break
    #     except Exception:
    #         continue

    return sample[0:pop]

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
            new_creacher[index] += rnd.random() -0.5
        return new_creacher

if __name__ == '__main__':
    sample = init_sample(d, ranges, pop)
    while (True):
        make_love(sample)
        length = len(sample)
        for index in range(length): sample.append(mutate(sample[index], mutate_mode = 1))
        sample = select(sample)        
        # output
        val = "{0:.3f}".format(f(sample[0]))
        dot = ["{0:.3f}".format(el) for el in sample[0]]
        print ('value: {} / {}'.format(val,dot))

