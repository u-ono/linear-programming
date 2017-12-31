#!/usr/bin/env python

import numpy as np
from numba import jit

@jit('f8[:](f8[:,:],f8[:],f8[:],f8[:],f8,f8)')
def affine_scaling(A, b, c, y0, gamma, eps):
    alfa = 0.0001
    y = y0
    while(1):
        s = c - np.dot(y, A)
        S_inv2 = np.diag(1.0 / (s*s))
        ASAT = np.dot(A, np.dot(S_inv2, A.T))
        delta_y = np.linalg.solve(ASAT, b)

        t = 0
        while np.all((c-np.dot((y+(t+alfa)*delta_y), A)) >= 0):
            t += alfa

        y_next = y + gamma*t*delta_y
        
        if np.dot(b, gamma*t*delta_y) <= eps:
            return y_next

        y = y_next
        
def main():
    A = np.array([[60.0, 40.0, 1.0, 0.0, 0.0],
                  [20.0, 30.0, 0.0, 1.0, 0.0],
                  [20.0, 10.0, 0.0, 0.0, 1.0]])
    b = np.array([3800.0, 2100.0, 1200.0])
    c = np.array([-400.0, -300.0, 0.0, 0.0, 0.0])

    y = np.array([-5.0, -5.0, -5.0])

    y_opt = affine_scaling(A, b, c, y, 0.5, 1.0e-2)

    for i in range(len(y_opt)):
        print("y{} : {}".format(i+1, round(y_opt[i])))
    print('Obj : {}'.format(round(np.dot(b, y_opt))))

if __name__ == "__main__":
    main()
