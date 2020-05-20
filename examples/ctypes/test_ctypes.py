#!/usr/bin/python3
from ctypes import *
import numpy as np

mylib = cdll.LoadLibrary('./libtestctypes.so')

ond_d_array = c_float(24)
two_d_mat = c_float(4.5)
two_d_mat = c_float(0)

mylib.add_float_p(ond_d_array, two_d_mat, byref(two_d_mat))
print('(1) {} + {} = {}'.format(ond_d_array.value, two_d_mat.value, two_d_mat.value))

# Specify argtypes
mylib.add_float_p.argtypes = [c_float, c_float, POINTER(c_float)]
mylib.add_float_p.restype = None
mylib.add_float_p(ond_d_array, two_d_mat, two_d_mat)
print('(2) {} + {} = {}'.format(ond_d_array.value, two_d_mat.value, two_d_mat.value))



#####################################
# Simmple 1-d array example
#####################################
ond_d_array = np.array([1,2,3,4,5]).astype(np.float32)
ptr = np.ctypeslib.ndpointer(np.float32, ndim=1, flags="C")

mylib.double_vector.argtypes = [ptr, c_uint]
# You can also set a return type
mylib.double_vector.restype = ptr
one_d_ret = mylib.double_vector(ond_d_array, ond_d_array.size)

print(ond_d_array, one_d_ret)


#####################################
# Even 2-d array is still represented as 1-d in c function
# 2-d representation is only a view of numpy.
#####################################
two_d_mat = np.array([[1,2,3,4,5], [6,7,8,9,10]]).astype(np.float32)
ptr_2d = np.ctypeslib.ndpointer(np.float32, ndim=2, flags="C")

mylib.double_vector.argtypes = [ptr_2d, c_uint]
mylib.double_vector(two_d_mat, two_d_mat.size)

print(two_d_mat)

#####################################
# Passing a shape as well as an array
#####################################

ndim = len(two_d_mat.shape)

# This is a way of creating primitive type array
shape = (c_int * ndim) (*two_d_mat.shape)

mylib.print.argtypes = [ptr_2d, c_int, c_int, c_int * ndim]
mylib.print(two_d_mat, two_d_mat.size, ndim, shape)

