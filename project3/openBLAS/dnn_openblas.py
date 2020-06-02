import os
import sys
import math
import networkx as nx
import numpy as np
from itertools import product
from multiprocessing import Process, sharedctypes
from ctypes import *

parallelism = 8
mylib = cdll.LoadLibrary('./openblas.so')

ptr = np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")
mylib.conv2D.argtypes = [c_int, c_int, c_int, c_int, c_int,
			 c_int, c_int, c_int, c_int, c_int,
			 ptr, ptr, ptr]
mylib.biasAdd.argtypes = [c_int, c_int, ptr, ptr, ptr]
mylib.maxPool2D.argtypes = [c_int, c_int, c_int, c_int,
		  	    c_int, c_int, c_int, 
			    ptr, ptr]
mylib.batchNorm.argtypes = [c_int, c_int, ptr, ptr, ptr, ptr, c_double, ptr]
mylib.leakyReLU.argtypes = [c_int, c_int, ptr, ptr]

class DnnInferenceEngine(object):
	def __init__(self, graph, debug):
		self.g = graph
		self.debug = debug

	def run(self, tin):
		self.g.in_node.set_input(tin)
		out = {}
		currents = [self.g.in_node]
		done = set()
		counter = 0
		if self.debug:
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
				current.run(counter)
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
	def __init__(self, name, in_node, kernel, strides, padding):
		self.name = name
		
		# input node
		self.in_node = in_node

		# weights
		self.weights = kernel.astype(np.float64, order = "C")
		assert len(self.weights.shape) == 4
		if len(self.in_node.result.shape) < 3:
			input_channels = 1
		else:
			input_channels = self.in_node.result.shape[-1]

		# strides
		if strides is None:
			strides = (1,1,1,1)
		assert len(strides) == len(self.in_node.result.shape)
		self.strides = strides

		# padding
		if padding == 'SAME':
			self.pad = (
					(0,0),
					(self.weights.shape[0]//2, self.weights.shape[0]//2),
					(self.weights.shape[1]//2, self.weights.shape[1]//2),
					(0,0)
					)
		elif padding == 'VALID':
			self.pad = ((0,0), (0,0), (0,0), (0,0))
		else:
			assert len(padding) == 2
			self.pad = padding
			
		ptin = np.pad(self.in_node.result, self.pad, mode='constant')
		self.PW = ptin.shape[1]
		self.PH = ptin.shape[2]
		self.KW = self.weights.shape[0]
		self.KH = self.weights.shape[1]
		self.IC = self.weights.shape[2]
		self.OC = self.weights.shape[3]
		self.SW = self.strides[1]
		self.SH = self.strides[2]
		self.OW = int((self.PW - self.KW) / self.SW + 1)
		self.OH = int((self.PH - self.KH) / self.SH + 1)

		self.result = np.zeros((1, self.OW, self.OH, self.OC)).astype(np.float64)

	def run(self, counter):
		ptin = np.pad(self.in_node.result, self.pad, mode='constant').astype(np.float64)
		
		mylib.conv2D(self.PW, self.PH, self.KW, self.KH, self.IC, self.OC, self.SW, self.SH, self.OW, self.OH, ptin, self.weights, self.result)
		# Assumed batch = 1		

class BiasAdd(DnnNode):
	def __init__(self, name, in_node, biases):
		self.name = name

		self.in_node = in_node 

		tin = self.in_node.result
		self.OW = tin.shape[1]
		self.OH = tin.shape[2]
		self.OC = tin.shape[3]

		self.biases = biases.astype(np.float64)
		assert self.biases.shape[-1] == self.OC 

		self.result = self.in_node.result.astype(np.float64) 

	def run(self, counter):
		tin = self.in_node.result.astype(np.float64)

		mylib.biasAdd(self.OW * self.OH, self.OC, tin, self.biases, self.result)
	
class MaxPool2D(DnnNode):
	def __init__(self, name, in_node, ksize, strides, padding):
		self.name = name

		# input node 
		self.in_node = in_node

		tin = self.in_node.result
		IW = tin.shape[1]
		IH = tin.shape[2]
		self.OC = tin.shape[3]

		# pooling kernel size
		assert len(ksize) == len(self.in_node.result.shape)
		self.ksize = ksize

		# stride
		self.stride = strides

		if padding == 'VALID':
			self.pad = (
					(0,0),
					(0,0),
					(0,0),
					(0,0))
		elif padding == 'SAME':
			w = self.in_node.result.shape[1]
			h = self.in_node.result.shape[2]

			out_w = math.ceil(float(w) / float(self.stride[1]))
			out_h = math.ceil(float(h) / float(self.stride[2]))
			pad_along_w = max(int((w - self.ksize[1]) / self.stride[1]) + 1 - w, 0)
			pad_along_h = max(int((h - self.ksize[2]) / self.stride[2]) + 1 - h, 0)
			pad_left = pad_along_w // 2
			pad_right = pad_along_w - pad_left
			pad_top = pad_along_h // 2
			pad_bottom = pad_along_h - pad_top
			self.pad = (
					(0,0),
					(pad_left,pad_right),
					(pad_top,pad_bottom),
					(0,0))
		else:
			raise Exception("Unexpected padding mode: {}".format(padding))	

		ptin= np.pad(self.in_node.result, self.pad, mode='constant')
		self.PW = ptin.shape[1]
		self.PH = ptin.shape[2]
		self.result = np.zeros((1, int(self.PW / self.stride[1]), int(self.PH / self.stride[2]), self.OC)).astype(np.float64)

	def run(self, counter):
		ptin = np.pad(self.in_node.result, self.pad, mode='constant').astype(np.float64)

		mylib.maxPool2D(self.PW, self.PH, self.ksize[1], self.ksize[2], self.OC, self.stride[1], self.stride[2], ptin, self.result)
	
class BatchNorm(DnnNode):
	def __init__(self, name, in_node, mean, variance, gamma, epsilon):
		self.name = name

		self.in_node = in_node

		tin = self.in_node.result
		self.OW = tin.shape[1]
		self.OH = tin.shape[2]
		self.OC = tin.shape[3]

		self.mean = mean.astype(np.float64) 
		assert self.mean.shape[0] == self.OC
		self.variance = variance.astype(np.float64)
		assert self.variance.shape[0] == self.OC 
		self.gamma = gamma.astype(np.float64)
		assert self.gamma.shape[0] == self.OC
		self.epsilon = np.float64(epsilon)
		
		self.result = self.in_node.result.astype(np.float64)

	def run(self, counter):
		tin = self.in_node.result.astype(np.float64)
		
		mylib.batchNorm(self.OW * self.OH, self.OC, tin, self.mean, self.gamma, self.variance, self.epsilon, self.result)
		
class LeakyReLU(DnnNode):
	def __init__(self, name, in_node):
		self.name = name

		self.in_node = in_node

		tin = self.in_node.result
		self.OW = tin.shape[1]
		self.OH = tin.shape[2]
		self.OC = tin.shape[3]

		self.result = self.in_node.result.astype(np.float64)

	def run(self, counter):
		tin = self.in_node.result.astype(np.float64)
			
		mylib.leakyReLU(self.OW * self.OH, self.OC, tin, self.result)
		
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

