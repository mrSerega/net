import math
import numpy as np
import binascii
import random


# config me here

word_length = 63
hem_power = 7
message = 'hello brand new world my friend'

# ^^^^^^^^^^^^^^

def bits2string(b=None):
    bb = []
    while len(b) > 7:
        bb.append(b[:7])
        b = b[7:]
    bb.append(b)
    return ''.join([chr(int(x, 2)) for x in bb])

def msg2bin(msg, bin_len):
    bin_msg = ' '.join(format(ord(x), 'b') for x in msg)
    print ('bin_msg: {}'.format(bin_msg))
    bin_msg = bin_msg.replace(' ','')
    ans = []
    while len(bin_msg) > 0:
        if len(bin_msg) < bin_len:
            bin_msg = bin_msg + '1'
            if len(bin_msg) >0:
                while(len(bin_msg)!=bin_len):
                    bin_msg = bin_msg+'0'
            ans.append(bin_msg)
            bin_msg = ''
            break
        else:
            ans.append(bin_msg[:bin_len])
            bin_msg = bin_msg[bin_len:]
    return ans

def hem_encode(bin_msg, code_power):
    for power in range(code_power):
        left = bin_msg[:2**power-1]
        right = bin_msg[2**power-1:]
        bin_msg = left+'0'+right
    bit_sum = np.zeros(code_power, dtype = int )
    for bit_index in range(len(bin_msg)):
        index = bit_index+1
        for power in range(code_power):
            if index % (2 ** (power+1)) > ((2 ** (power+1)) / 2 - 1) and index % (2 ** (power+1)) < (2 ** (power+1)):
                # print (bit_index)
                if bin_msg[bit_index] == '1' : bit_sum[power]+=1
    # print(bit_sum)
    msg_arr = list(bin_msg)
    for s_index in range(len(bit_sum)):
        msg_arr[2**s_index-1] = str(bit_sum[s_index] % 2)
    b0 = 0
    for el in msg_arr:
        if el == '1':
            b0 +=1
    msg_arr.append(str(b0 % 2))
    return msg_arr

def hem_decode(hem_msg, code_power):
    b0 = 0
    for el in hem_msg[:-1]:
        if el == '1':
            b0 +=1
    bit_sum = np.zeros(code_power, dtype = int )
    tmp_msg = hem_msg[:]
    tmp_msg[-1] = str(b0 % 2)
    for p in range(code_power):
        tmp_msg[2**p-1] = '0'
    for bit_index in range(len(tmp_msg[:-1])):
        index = bit_index+1
        for power in range(code_power):
            if index % (2 ** (power+1)) > ((2 ** (power+1)) / 2 - 1) and index % (2 ** (power+1)) < (2 ** (power+1)):
                if tmp_msg[bit_index] == '1' : bit_sum[power]+=1
    msg_arr = tmp_msg[:]
    for s_index in range(len(bit_sum)):
        msg_arr[2**s_index-1] = str(bit_sum[s_index] % 2)
    
    to_invert = -1
    
    # print ('msg_arr:\n{}'.format(msg_arr))

    for index in range(len(msg_arr[:-1])): 
        if msg_arr[index] != hem_msg[index]: to_invert += (index+1)

    if to_invert == -1 and hem_msg[-1] == tmp_msg[-1]:
        # print ('no error')
        return hem_msg

    if hem_msg[-1] != tmp_msg[-1] and to_invert != -1: 
        # print (to_invert)
        if to_invert < len(hem_msg): 
            msg_arr = hem_msg[:]
            msg_arr[to_invert] = str(int(not bool(hem_msg[to_invert])))
            return msg_arr

    if to_invert != -1 and hem_msg[-1] == tmp_msg[-1]:
        print ('Double error!')
        return hem_msg
    else:
        print ('cant fix')
        return hem_msg

def bin2ascii(message):
    msg = message[:]
    while msg[-1] == '0':
        msg.pop(-1)
    msg.pop(-1)
    msg = ''.join(msg)
    return bits2string(b=msg)

def delCodeBits(data, code_power):
    powers = list(range(code_power))
    powers.reverse()
    for power in powers:
        data.pop(2**power-1)
    data.pop(-1)

def doMistale(package, number_of_mistake):
    for mistake in range(number_of_mistake):
        index = int(random.uniform(0,len(package)))
        package[index] = str(int(not bool(package[index])))

if __name__ == '__main__':
    message = message.replace(' ','_')
    bin_msg = msg2bin(message, word_length)
    newmsg = []
    for word in bin_msg: 
        hc = hem_encode(word, hem_power)
        print (hc)
        doMistale(hc, 1)
        hd = hem_decode(hc, hem_power)
        # hd = hc
        delCodeBits(hd, hem_power)
        newmsg.extend(hd)
    recived = bin2ascii(newmsg).replace('_',' ')
    print (recived)

