from ctypes import *
import numpy as np

mylib = cdll.LoadLibrary('./matmul.so')

I = np.arange(0,12,2)
I = np.reshape(I, (2,3))
W = np.arange(6,18,2)
W = np.reshape(I, (3,2))
O = np.zeros((2,2))

IL = np.reshape(I, 2*3)
IC = np.ctypeslib.as_ctypes(IL)
WL = np.reshape(W, 3*2)
WC = np.ctypeslib.as_ctypes(WL)
OL = np.reshape(O, 2*2)
OC = np.ctypeslib.as_ctypes(OL)

print(OL)
mylib.matmul.argtypes = [c_int, c_int, c_int, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ndpointer(ctypes.c_double)]
mylib.matmul(2,2,3,IL,WL,OL)

print(OL)
