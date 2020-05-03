import os
import sys
import math
import networkx as nx
import numpy as np

class DnnInferenceEngine(object):
	def __init__(self, graph):
		self.g = graph

	def run(self, tin):
		self.g.in_node.set_input(tin)
		out = {}
		currents = [self.g.in_node]
		done = set()
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
				current.run()
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
		return self.out_node is node

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
	'''
	This is an interface for layers
	'''
	def __init__(self):
		'''
		Construct the layer and print its name if correctly built
		'''
		self.in_node = None	# type: np.ndarray
		pass

	def run(self):
		'''
		After invoking `DnnNode.run` method, `DnnNode.result` will contin the calculated value
		'''
		self.result = None	# type: np.ndarray
	
	def _notify_completion(self, name: str):
		'''
		Print the given name.
		Note that this is a private method and also it must be invoked only when construction is completed.
		'''
		print(name)

	def _verify_shapes(self, lhs: np.ndarray, rhs: np.ndarray):
		'''
		Compare the shapes of given numpy arrays.
		If the shape is not equal, then TypeError would be raised to interrupt the flow.

		@throws TypeError
		'''
		if lhs.shape != rhs.shape:
			className = self.__class__.__name__
			raise TypeError("While constructing {}, lhs's shape {} != rhs's shape {}".format(className, lhs.shape, rhs.shape))


#
# Complete below classes.
#

class Conv2D(DnnNode):
	def __init__(self, name: str, in_node: DnnNode, 
			kernel: np.ndarray, strides: list, padding: str):
		'''

		@param ("SAME" | "VALID") padding
		'''

		self.in_node = in_node
		self.kernel = kernel
		self.strides = strides




		self._notify_completion(name)

	def run(self):
		pass

	def _stride(self):

	def _pad(self, matrix: np.ndarray) -> np.ndarray:

		return None

class BiasAdd(DnnNode):
	def __init__(self, name: str, in_node: DnnNode, biases: np.ndarray):
		self._verify_shapes(in_node.result, biases)
		self.in_node, self.biases = in_node, biases
		self.result = None

		self._notify_completion(name)

	def run(self):
		self.result = self.in_node.result + self.biases

class MaxPool2D(DnnNode):
	def __init__(self, name, in_node, ksize, strides, padding):
		pass
		
	def run(self):
		pass

class BatchNorm(DnnNode):
	def __init__(self, name, in_node, mean, variance, gamma, epsilon):
		pass

	def run(self):
		pass

class LeakyReLU(DnnNode):
	def __init__(self, name: str, in_node: DnnNode):
		self.in_node = in_node

		self._notify_completion(name)

	def run(self):
		value = self.in_node.result
		self.result = np.maximum(value, 0.1 * value)



# Do not modify below
class Input(DnnNode):
	def __init__(self, name, in_shape):
		self.name = name
		self.in_shape = in_shape 
		self.result = np.ndarray(self.in_shape)

	def set_input(self, tensor):
		assert tuple(self.in_shape) == tuple(tensor.shape)
		self.result = tensor 

	def run(self):
		pass

