#!/usr/bin/env python

import pickle
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit('f8[:](f8[:,:],f8[:],f8[:],f8[:],f8,f8)')
def affine_scaling(A, b, c, y0, gamma, eps):
    alfa = 1.0e-5

    prime_obj_list = [np.dot(b, y0)]
    dual_obj_list = []
    x_list = []
    y_list = [y0]

    y = y0
    while(1):
        s = c - np.dot(y, A)
        S_inv2 = np.diag(1.0 / (s*s))
        ASAT = np.dot(A, np.dot(S_inv2, A.T))
        delta_y = np.linalg.solve(ASAT, b)
        x = np.dot(S_inv2, np.dot(A.T, delta_y))
        x_list.append(x)

        obj = np.dot(c, x)
        dual_obj_list.append(obj)
        print('dual obj_func value : {}'.format(round(obj)))

        t = 0
        while np.all((c-np.dot((y+(t+alfa)*delta_y), A)) >= 0):
            t += alfa

        y_next = y + gamma*t*delta_y
        y_list.append(y_next)
        
        obj = np.dot(b, y)
        prime_obj_list.append(obj)
        print('primary obj_func value : {}'.format(round(obj)))

        if np.dot(b, gamma*t*delta_y) <= eps:
            s = c - np.dot(y, A)
            S_inv2 = np.diag(1.0 / (s*s))
            ASAT = np.dot(A, np.dot(S_inv2, A.T))
            delta_y = np.linalg.solve(ASAT, b)
            x = np.dot(S_inv2, np.dot(A.T, delta_y))
            x_list.append(x)

            obj = np.dot(c, x)
            dual_obj_list.append(obj)
            print('dual obj_func value : {}'.format(round(obj)))

            return y_list, x_list, prime_obj_list, dual_obj_list

        y = y_next


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

    return affine_scaling(a, b, c, y0, gamma, eps)

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

    return affine_scaling(a, b, c, y0, gamma, eps)

def main():
    
    gamma = 0.66
    eps = 0.5

    y_list, x_list, prime_obj_list, dual_obj_list = problem_4(gamma, eps)
    rep = np.arange(len(y_list))

    obj_opt = (prime_obj_list[-1] + dual_obj_list[-1]) / 2.0

    fig1 = plt.figure(1)

    ax1 = fig1.add_subplot(211)
    ax1.plot(rep, np.log10(np.absolute(prime_obj_list - obj_opt)))
    ax1.grid(True)
    ax1.set_xlabel('repetition')
    ax1.set_ylabel('log10(error)')
    ax1.set_title('Primary Objective Function Error')

    ax2 = fig1.add_subplot(212)
    ax2.plot(rep, np.log10(np.absolute(dual_obj_list - obj_opt)))
    ax2.grid(True)
    ax2.set_xlabel('repetition')
    ax2.set_ylabel('log10(error)')
    ax2.set_title('Dual Objective Function Error')

    fig1.tight_layout()
    plt.show()

    y_opt = np.round(100*y_list[-1])
    print(y_opt)
