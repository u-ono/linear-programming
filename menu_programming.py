#!/usr/bin/env python

import pickle
import numpy as np
from numba import jit

@jit('f8[:](f8[:,:],f8[:],f8[:],f8[:],f8,f8)')
def affine_scaling(A, b, c, y0, gamma, eps):
    alfa = 1.0e-5
    obj_list = []
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
            obj = np.dot(b, y)
            obj_list.append(obj)
            return y_next, obj_list

        y = y_next

        obj = np.dot(b, y)
        obj_list.append(obj)
        print('objective function value : {}'.format(round(obj)))

def problem_3(gamma, eps):

    f = open('matrix.pkl', 'rb')
    a_tilde, b_tilde, c_tilde = pickle.load(f)
    f.close()

    m = b_tilde.shape[0]
    n = c_tilde.shape[1]
    b = -b_tilde
    c = -np.hstack((c_tilde[0, :], np.zeros(m)))
    a = -np.hstack((a_tilde[:,:-1], np.diag(np.ones(m))))

    y0 = np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.])

    y_opt, obj_list = affine_scaling(a, b, c, y0, gamma, eps)

    print('y_opt : {}'.format(y_opt))

    return y_opt, obj_list

def problem_4(gamma, eps):

    f = open('matrix.pkl', 'rb')
    a_tilde, b_tilde, c_tilde = pickle.load(f)
    f.close()

    m = b_tilde.shape[0]
    n = c_tilde.shape[1]
    b = -b_tilde
    c_tilde = np.append(c_tilde, np.array([-200., -200.]).reshape(2, 1), axis=1)
    c = -np.hstack((c_tilde[0, :], np.zeros(m)))
    a_tilde[:,-1] *= -1
    a = -np.hstack((a_tilde, np.diag(np.ones(m))))

    y0 = np.array([10., 10., 10., 10., 0.1, 0.1, 0.1, 0.1, 1.5, 0.1, 10., 0.1, 0.1, 0.1, 10., 10., 0.1, 10., 10., 0.1, 10.])

    y_opt, obj_list = affine_scaling(a, b, c, y0, gamma, eps)

    print('y_opt : {}'.format(y_opt))

    return y_opt, obj_list

def main():
    
    gamma = 0.66
    eps = 0.5

    y_opt, obj_list = problem_3(gamma, eps)

    obj_opt = obj_list[-1]
