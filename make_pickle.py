#!/usr/bin/env python

import pickle
import numpy as np

def load_file(name):
    lines = open(name).readlines()
    a = []
    b = []
    c = []

    lines_c = lines[-2:]
    for line in lines_c:
        lst = line[:-1].split()
        tmp_c = []
        for i in range(len(lst)):
            tmp_c.append(float(lst[i]))
        c.append(tmp_c)
    del lines[-2:]

    for line in lines:
        lst = line[:-1].split()
        tmp_a = []
        tmp_b = 0
        for i in range(len(lst)):
            if i == len(lst)-1:
                tmp_b = float(lst[i])
            else:
                tmp_a.append(float(lst[i]))
        a.append(tmp_a)
        b.append(tmp_b)
    return np.array(a), np.array(b), np.array(c)

def main():
    a, b, c = load_file('kondate171130.txt')
    f = open('matrix.pkl', 'wb')
    pickle.dump([a, b, c], f)
    f.close

if __name__ == '__main__':
    main()
