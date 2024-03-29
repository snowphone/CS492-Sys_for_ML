import os
import sys
import math
import networkx as nx
import numpy as np
from numpy.ctypeslib import ndpointer
from itertools import product
from multiprocessing import Process, sharedctypes

from ctypes import *

lib = cdll.LoadLibrary("./libcudadnn.so")

parallelism = 8

class DnnInferenceEngine(object):
	def __init__(self, graph, debug=False):
		self.g = graph
		self.debug = debug

	def run(self, tin):
		self.g.in_node.set_input(tin)
		out = {}
		currents = [self.g.in_node]
		done = set()
		counter = 0
		os.makedirs("intermediate", exist_ok=True)
		while (len(currents) != 0):
			nexts = []
			for current in currents:
				skip_current = False
				predecessors = self.g.G.predecessors(current)
				for predecessor in predecessors:
					if predecessor not in done:
						nexts.append(predecessor)
						skip_current = True
				if skip_current:
					continue
				
				print("Start running Layer: {}, counter: {}".format(current.name, counter))
				current.run(counter)
				print("Done")
				if not isinstance(current, Input):
					if self.debug:
						path = os.path.join("intermediate", "layer_{}.npy".format(counter))
						np.save(path, current.result)
					counter += 1
				if self.g.is_out_node(current):
					out = current.result
				done.add(current)
				for successor in self.g.G.successors(current):
					nexts.append(successor)
			currents = nexts
		return out

class DnnGraphBuilder(object):
	def __init__(self):
		self.G = nx.DiGraph()
		self.name_num = {"conv2d": 0, 
						 "bias_add": 0, 
						 "max_pool2d": 0, 
						 "batch_norm": 0, 
						 "leaky_relu": 0, 
						 "input": 0}
		self.in_node = None
		self.out_node = None

	def set_in_node(self, node):
		self.in_node = node

	def set_out_node(self, node):
		self.out_node = node

	def is_out_node(self, node):
		if self.out_node is node:
			return True
		else:
			return False

	def get_name(self, layer_name):
		name = layer_name + "_" + str(self.name_num[layer_name])
		self.name_num[layer_name] += 1
		return name

	def create_conv2d(self, in_node, kernel, strides, padding):
		out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
		self.G.add_edge(in_node, out_node)
		return out_node

	def create_bias_add(self, in_node, biases):
		out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
		self.G.add_edge(in_node, out_node)
		return out_node

	def create_max_pool2d(self, in_node, ksize, strides, padding):
		out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
		self.G.add_edge(in_node, out_node)
		return out_node

	def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
		out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
		self.G.add_edge(in_node, out_node)
		return out_node

	def create_leaky_relu(self, in_node):
		out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
		self.G.add_edge(in_node, out_node)
		return out_node

	def create_input(self, in_shape):
		out_node = Input(self.get_name("input"), in_shape)
		self.G.add_node(out_node) 
		self.set_in_node(out_node)  # Assume there's only one input
		return out_node

class DnnNode(object):
	def __init__(self):
		pass

	def run(self, counter):
		self.result = None 

class Conv2D(DnnNode):
	def __init__(self, name, in_node, kernel: np.ndarray, strides, padding):
		self.name = name
		
		# input node
		self.in_node = in_node


		# strides
		if strides is None:
			strides = (1,1,1,1)
		assert len(strides) == len(self.in_node.result.shape)
		self.strides = strides

		# padding
		assert padding.upper() in {"VALID", "SAME"}
		self.padding = padding
			

		self.kernel = kernel.astype(np.float32, order="C")
		assert len(self.kernel.shape) == 4


		self.conv2d = lib.conv2d
		ptr_t = ndpointer(dtype=np.float32)
		shape_t = (c_int * 4)
		kernel_t = ptr_t
		stride_t = (c_int * 2)
		padding_t = c_int

		self.conv2d.argtypes = [ptr_t, shape_t, kernel_t, shape_t, stride_t, padding_t]

		self.result = np.empty(self.calc_new_shape())
		return

	def calc_new_shape(self):
		i = self.in_node.result
		n_batch = i.shape[0]
		n_row = i.shape[1]
		n_col = i.shape[2]
		if self.padding.upper() == "VALID":
			n_row = (n_row - self.kernel.shape[0]) // self.strides[1] + 1
			n_col = (n_col - self.kernel.shape[1]) // self.strides[2] + 1

		n_chan = self.kernel.shape[3]
		return (n_batch, n_row, n_col, n_chan)


	def run(self, counter):
		print("Start of layer: {}".format(self.name))


		
		t_in = self.in_node.result.astype(np.float32, order='C')
		shape = (c_int * 4) (*t_in.shape)
		k = self.kernel
		k_shape = (c_int * 4) (*k.shape)
		s = (c_int * 2) (*self.strides[1:3])
		padding_enum = 0 if self.padding.upper() == "VALID" else 1

		self.conv2d.restype = ndpointer(dtype=np.float32, shape=self.calc_new_shape())

		self.result = self.conv2d(t_in, shape, k, k_shape, s, padding_enum)

		return


class BiasAdd(DnnNode):
	def __init__(self, name, in_node, biases):
		self.name = name

		self.in_node = in_node 

		tin = self.in_node.result
		self.OW = tin.shape[1]
		self.OH = tin.shape[2]
		self.OC = tin.shape[3]

		self.biases = biases 
		assert self.biases.shape[-1] == self.OC 

		self.result = self.in_node.result 
		self.bias_add = lib.bias_add
		ptr_t = ndpointer(np.float32)
		self.ndim = len(tin.shape)
		self.bias_add.argtypes = [ptr_t, c_int * self.ndim, ptr_t]


	def run(self, counter):
		self.result = self.in_node.result.astype(np.float32)
		shape = (c_int * self.ndim) (*self.result.shape)
		self.bias_add(self.result, shape, self.biases)

		return


class MaxPool2D(DnnNode):
	def __init__(self, name, in_node, ksize, strides, padding):
		self.name = name

		# input node 
		self.in_node = in_node

		# pooling kernel size
		assert len(ksize) == len(self.in_node.result.shape)
		self.ksize = ksize

		# stride
		self.stride = strides

		# padding
		self.padding = padding
		assert padding in {"VALID", "SAME"}


		###################
		##### MY CODE #####
		###################

		self.maxpool = lib.maxpool
		input_t = np.ctypeslib.ndpointer(dtype=np.float32)
		ndim = len(self.in_node.result.shape)
		shape = (c_int * ndim)
		ksize_t = (c_int * 2)
		stride_t = (c_int * 2)
		padding_t = c_int
		self.maxpool.argtypes = [input_t, shape, ksize_t, stride_t, padding_t]

		self.result = np.empty(self.calc_outshape())

	def calc_outshape(self):

		def calc_new_n(n: int, k: int, s: int):
			return (n - k) // s + 1

		def calc_pad(n: int, k: int, s: int):
			if n % s:
				return max(k - (n % s), 0)
			else:
				return max(k - s, 0)

		n_batch, r, c, n_chan = self.in_node.result.shape

		if self.padding.upper() == "SAME":
			r += calc_pad(r, self.ksize[1], self.stride[1])
			c += calc_pad(c, self.ksize[2], self.stride[2])
		
		new_r = calc_new_n(r, self.ksize[1], self.stride[1])
		new_c = calc_new_n(c, self.ksize[2], self.stride[2])

		shape = (n_batch, new_r, new_c, n_chan)
		return shape

	def run(self, counter):

		# My code
		t_in = self.in_node.result
		ndim = len(t_in.shape)
		shape = (c_int * ndim) (*t_in.shape)
		k = (c_int * 2) (*self.ksize[1:3])
		s = (c_int * 2) (*self.stride[1:3])
		p = 0 if self.padding.upper() == "VALID" else 1

		self.maxpool.restype = np.ctypeslib.ndpointer(dtype=c_float, shape=self.calc_outshape())

		self.result = self.maxpool(t_in, shape, k, s, p)

		return

class BatchNorm(DnnNode):
	def __init__(self, name, in_node, mean, variance, gamma, epsilon):
		self.name = name

		self.in_node = in_node

		tin = self.in_node.result
		self.OW = tin.shape[1]
		self.OH = tin.shape[2]
		self.OC = tin.shape[3]

		self.mean = mean 
		assert self.mean.shape[0] == self.OC
		self.variance = variance
		assert self.variance.shape[0] == self.OC 
		self.gamma = gamma
		assert self.gamma.shape[0] == self.OC
		self.epsilon = epsilon
		
		self.result = self.in_node.result

		self.batch_normalization = lib.batch_normalization
		ptr_t = ndpointer(np.float32)
		ndim = len(tin.shape)
		self.batch_normalization.argtypes = [ptr_t, c_int * ndim, ptr_t, ptr_t, ptr_t, c_float]

	def run(self, counter):
		self.result = self.in_node.result.astype(np.float32)

		shape = (c_int * len(self.result.shape)) (*self.result.shape)
		mean = self.mean.astype(np.float32)
		var = self.variance.astype(np.float32)
		gamma = self.gamma.astype(np.float32)
		epsilon = self.epsilon

		self.batch_normalization(self.result, shape, mean, var, gamma, epsilon)


class LeakyReLU(DnnNode):
	def __init__(self, name, in_node):
		self.name = name

		self.in_node = in_node
		self.result = in_node.result

		self.leaky_relu = lib.leaky_relu
		pointer_t = ndpointer(np.float32, ndim=4)
		self.leaky_relu.argtypes = [pointer_t, c_int]

		return



	def run(self, counter):
		self.result = self.in_node.result.astype(np.float32)
		self.leaky_relu(self.result, self.result.size)
		return


class Input(DnnNode):
	def __init__(self, name, in_shape):
		self.name = name
		self.in_shape = in_shape 
		self.result = np.ndarray(self.in_shape)

	def set_input(self, tensor):
		assert tuple(self.in_shape) == tuple(tensor.shape)
		self.result = tensor 

	def run(self, counter):
		pass


