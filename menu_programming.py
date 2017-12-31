#!/usr/bin/env python

import pickle
import numpy as np
from numba import jit

@jit('f8[:](f8[:,:],f8[:],f8[:],f8[:],f8,f8)')
def affine_scaling(A, b, c, y0, gamma, eps):
    alfa = 1.0e-5
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

def problem_3():

    f = open('matrix.pkl', 'rb')
    a_tilde, b_tilde, c_tilde = pickle.load(f)
    f.close()

    m = b_tilde.shape[0]
    n = c_tilde.shape[1]
    b = -b_tilde
    c = -np.hstack((c_tilde[0, :], np.zeros(m)))
    a = -np.hstack((a_tilde[:,:-1], np.diag(np.ones(m))))

    y_tilde = np.array([100., 100., 100., 200., 100., 100., 50., 50., 200., 50., 50., 70., 80., 100., 60., 50., 80., 90., 50., 90., 100.])

    y_opt = affine_scaling(a, b, c, y_tilde, 0.66, 1.0e-1)

    print('y_opt : {}'.format(y_opt))
