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

mode = 'eggholder'

pop = 100        #population amount                                         
gen_len = 64      #(do not config)
p = 0.50           #mutation 

if mode == 'booth':
    d = 2                                #demenision of var    (do not config)                                      
    ranges = [[-10, 10], [-10, 10]]      #(do not config)
    def f(v):
        return float((decimal.Decimal(v[0])+decimal.Decimal(2*v[1])-7)**2 + (decimal.Decimal(2*v[0])+decimal.Decimal(v[1])-5)**2)       #ranges for each dem
elif mode == 'rosenbrock':
    d = 6                                      #demenision of var  
    ranges = [[-10,10] for el in range(d)] 
    def f(v):
        s = 0
        for i in range(d-1):
            s += float(100 * (decimal.Decimal(v[i+1])-decimal.Decimal(v[i])**2)**2 + decimal.Decimal((1-v[i]))**2)
        return s
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



def cross(mom, dad):
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
    
    


def make_love(population):
    rnd.shuffle(population)
    l = len(population)
    for index in range(int(l/2)):
        siblings = cross(population[index],population[l-1-index])
        population.append(siblings[0])
        population.append(siblings[1])

def select(population):
    sample.sort(key=f)
    # print (population)
    return sample[0:pop]

def mutate(creacher):
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

if __name__ == '__main__':
    sample = init_sample(d, ranges, pop)
    while (True):
        make_love(sample)
        length = len(sample)
        for index in range(length): sample.append(mutate(sample[index]))
        sample = select(sample)        
        print ('value: {} / {}\r'.format(f(sample[0]),sample[0]))

