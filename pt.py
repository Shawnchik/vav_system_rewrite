# -*- coding: utf-8 -*-
import numpy as np
import itertools

w = np.zeros((10,10))

a = [[[1,1],[1,0]],
     [[1,0],[1,1]],
     [[1,1],[1,1]],
     [[1,1],[1,0],[1,0]],
     [[1,0],[1,1],[1,0]],
     [[1,1,1],[0,0,1]],
     [[1,1],[1,0]],
     [[1,1],[1,1]],
     [[1,1,1,1]],
     [[1,0,0],[1,1,1]],
     [[0,1,0],[1,1,1],[0,1,0]],
     [[1,1,1]],
     [[1,0,0],[1,1,1]],
     [[1,1],[1,1]],
     [[1],[1],[1]],
     [[1,1],[1,1]],
     [[0,1,0],[1,1,1],[0,1,0]],
     [[1,0],[1,1]],
     [[0,1,0],[1,1,1]],
     [[1,0],[1,1]],
     [[0,1,0],[1,1,1],[0,1,0]],
     [[0,1],[1,1],[0,1]],
     [[1,1],[1,0],[1,0]],
     [[1,1,1],[1,0,0]],
     [[1,0,0],[1,1,1]],
     [[0,1],[0,1],[1,1]]]
# print([np.multiply(a[i], i+1) for i in range(len(a))])
for i in range(len(a)):
	print(np.multiply(a[i], i+1))
a = [np.multiply(a[i], i+1) for i in range(len(a))]
'''
b = [[[1,1],[1,1]], [[0,1,0],[1,1,1],[0,1,0]],[[0,1],[0,1],[1,1]],[[1,0,0],[1,1,1]],[[1],[1],[1],[1]],[[1],[1]],[[1,1]]]
for i in range(len(b)):
	print(np.multiply(b[i], i+1))
b = ([np.multiply(b[i], i+1) for i in range(len(b))])
'''

def i_t(w, i, j, a):
    found = 0
    for k in range(len(a)):
        for l in range(len(a[k])):
            if a[k][l] > 0:
                if w[i + k][j + l] == 0:
                    found = 1
                else:
                    found = 0
                    return False
    if found == 1:
        return True

def i_w(w, a):
    for i in range(len(w)-len(a)+1):
        for j in range(len(w[i])-len(a[0])+1):
            if i_t(w, i, j, a):
                for k in range(len(a)):
                    for l in range(len(a[k])):
                        if a[k][l] > 0:
                            w[i+k][j+l] = a[k][l]
                # print(w)
                return w, True
    return w, False

# abcd = [a,b,c,d,e,f,g]
# index = list(itertools.permutations(range(7),7))

for i in range(200000):
	index = np.random.permutation(24)
	# print(index)
	w = np.zeros((10,10))
	found = 1
	for j in range(24):
	    if found:
	        w, found = i_w(w, a[index[j]])
	        # print(w)
	    else:
	        break
	if found:
	    print(index)
	    print(np.array(w))


# index = list(itertools.permutations(range(8),8))
# print(len(index))




