# -*- coding: utf-8 -*-
import numpy as np
import itertools
'''
w = np.zeros((4,4))
a = np.zeros((2,2)) + 1
b = np.zeros((3,2))
b[0,1] = b[1, 1] =b[2,0]=b[2,1]=2
c = np.zeros((2,3))
c[0,0]=c[1,0]=c[1,1]=c[1,2]=3
d = np.zeros((4,1)) + 4
print(w)
print(a)
print(b)
print(c)
print(d)

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

abcd = [a,b,c,d]
index = list(itertools.permutations([0, 1, 2, 3],4))

for i in range(len(index)):
    w = np.zeros((4, 4))
    found = 1
    for j in range(4):
        if found:
            w, found = i_w(w, abcd[index[i][j]])
        else:
            break
    if found:
        print(index[i])
        print(w)
'''

index = list(itertools.permutations(range(12),12))
print(len(index))




