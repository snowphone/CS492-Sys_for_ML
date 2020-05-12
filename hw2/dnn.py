import os
import sys
import numpy as np
import math
import networkx as nx

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

class DNNException(AssertionError):
	def __init__(self, msg):
		super().__init__(msg)

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


	def _check_quadruple(self, quadruple):
		'''
		A quadruple (ksize or stride) must be formed as (1, x, x, 1)

		@param quadruple A list or tuple of four integers. First and last element must be 1 and two central elements must be equal.
		'''
		batch, height, width, channel = quadruple
		if not (batch == channel and height == width):
			raise DNNException()


	def _check_padding(self, option: str):
		if option.upper() not in {"SAME", "VALID"}:
			raise DNNException("padding argument must be one of 'SAME' or 'VALID', not {}".format(option))


	@staticmethod
	def _make_batch_major(chan_last: np.ndarray):
		chan_last = np.array(chan_last)
		last_dim = len(chan_last.shape) - 1
		first_dim = 0
		return np.moveaxis(chan_last, last_dim, first_dim)

	@staticmethod
	def _make_batch_minor(chan_first: np.ndarray):
		chan_first = np.array(chan_first)
		last_dim = len(chan_first.shape) - 1
		first_dim = 0
		return np.moveaxis(chan_first, first_dim, last_dim)

	@staticmethod
	def _make_channel_last(chan_first: np.ndarray, consider_batch=True)->np.ndarray:
		chan_first = np.array(chan_first)
		last_dim = len(chan_first.shape) - 1
		first_dim = 1 if consider_batch else 0
		return np.moveaxis(chan_first, first_dim, last_dim)

	@staticmethod
	def _make_channel_first(chan_last: np.ndarray, consider_batch=True) -> np.ndarray:
		chan_last = np.array(chan_last)
		last_dim = len(chan_last.shape) - 1
		first_dim = 1 if consider_batch else 0
		return np.moveaxis(chan_last, last_dim, first_dim)

	@staticmethod
	def _calc_same_pad(n: int, ksize: int, stride: int) -> int:
		'''
		Calculate the size of padding of maxpool2d when "SAME" options is set.
		'''
		if n % stride:
			return np.max(ksize - (n % stride), 0)
		else:
			return np.max(ksize - stride, 0)


	@staticmethod
	def _stride(matrix: np.ndarray, ksize: int, stride: int, pad_mode="constant")->np.ndarray:
		'''
		TODO: Parallelize!

		@return np.ndarray shape: (batch, out_n x out_n, tile)  => (batch, n*n, f, f, c)
		Tile means a local area of the matrix and its shape equals to the kernel.
		'''


		row, col = matrix.shape[1:3]
		r_pad = DnnNode._calc_same_pad(row, ksize, stride)
		c_pad = DnnNode._calc_same_pad(col, ksize, stride)
		if pad_mode == "edge":
			pad = [(0, r_pad), (0, c_pad)]
			matrix = DnnNode._pad(matrix, ksize, stride, pad_mode, pad=pad)

		matrix = DnnNode._make_batch_minor(matrix)


		
		calc_padded_len = lambda x: (x - ksize) // stride + 1
		padded_row, padded_col = matrix.shape[:2]


		tiles = np.array([
			matrix[r:r+ksize, c:c+ksize]
			for r in range(0, row, stride) 
			for c in range(0, col, stride)
			if r != row and c != col and 
				r + ksize <=padded_row and c + ksize <= padded_col
			])

		tiles = DnnNode._make_batch_major(tiles)

		return tiles

	@staticmethod
	def _pad(matrix: np.ndarray, ksize: int, 
			n_stride: int, mode="constant", pad=None) -> np.ndarray:
		'''
		Insert a padding for the purpose of keeping the output's shape same.
		For example, if the input is 3 x 5 shaped, then the result with padding is also 3 x 5.

		@params np.ndarray matrix 
		@params int ksize The length of the filter. Assume that # of rows 
				in the filter equals to # of the columns of the filter. 
				In short, the filter is regarded as square.
		@params int n_stride the stride step. It also assumes vertical step == horizontal step.
		@params mode "constant" or "edge". 
				If 'constant' is set, padding will be filled with zeroes. 
				It is normally used for convolutional layers.
				If 'edge' is set, padding will be filled with edge values.
				It is used for pooling, especially maxpooling, not to contaminate the values.
		@params pad [(r_pad, r_pad), (c_pad, c_pad)] 
				Only two dimentions (row, column) are need since other dimentions are handled inside the code. 
				So, you don't need to think about other dimentions.
				A tuple consists of two elements. The first value is for 'before' padding, 
				and the other is for 'after' padding.
				For example, if (bef, aft) tuple is given, 
				then a vector will be padded like 'bef' 'vector' 'after'.
				If pad is set, then ksize and n_stride are ignored.
		'''
		matrix = DnnNode._make_batch_minor(matrix)
		if pad is None:
			n_row, n_col = matrix.shape[:2]
			v_pad = ((n_stride - 1) * n_row - 1 + ksize) // 2
			h_pad = ((n_stride - 1) * n_col - 1 + ksize) // 2

			pad = [(v_pad, v_pad), (h_pad, h_pad)]	# [(up, down), (left, right)]

		pad = [*pad, (0, 0), (0, 0)]	# First zeroes are for channels, seconds are for batches

		matrix = np.pad(matrix, pad_width=pad, mode=mode)

		matrix = DnnNode._make_batch_major(matrix)
		return matrix


class Conv2D(DnnNode):
	def __init__(self, name: str, in_node: DnnNode, 
			kernels: np.ndarray, strides: list, padding: str):
		'''
		@param padding Note that "SAME" or "VALID" (case insensitive) can be permitted.
		'''
		if in_node.result.shape[-1] != kernels.shape[-2]:
			raise DNNException("the number of output channels is different")
		self._check_quadruple(strides)
		self._check_padding(padding)

		self.ksize = kernels.shape[0]
		self.stride = strides[1]
		self.in_node = in_node
		self.kernels = kernels
		self.padding = padding
		if(padding.upper() == "SAME"):
			out_w = math.floor(in_node.result.shape[1]/strides[1]) + 1
			out_h = math.floor(in_node.result.shape[2]/strides[2]) + 1
			self.result = np.ndarray((in_node.result.shape[0], out_w, out_h, kernels.shape[3]))
		else:
			out_w = math.floor((in_node.result.shape[1] - kernels.shape[0])/strides[1]) + 1
			out_h = math.floor((in_node.result.shape[2] - kernels.shape[1])/strides[2]) + 1
			self.result = np.ndarray((in_node.result.shape[0], out_w, out_h, kernels.shape[3]))
		
		self.name = name
		self._notify_completion(name)

	def run(self):
		'''
		Strategy: (outchan x kernel) @ (tile x strides).
		Each operand is a matrix and therefore kernel.length == tile.length matmul can be applied!
		'''

		matrix = self.in_node.result

		if self.padding.upper() == "SAME":
			matrix = self._pad(matrix, self.ksize, self.stride)


		n_outchan = self.kernels.shape[-1]
		w = self._make_channel_first(self.kernels, consider_batch=False)
		w = w.reshape(n_outchan, -1)	# (out_chan, f * f * c)


		n_batch = matrix.shape[0]
		x = self._stride(matrix, self.ksize, self.stride)	# (n_batch, out_row * out_col, f, f, c)
		x = x.reshape(n_batch, x.shape[1], -1)				# (n_batch, out_row * out_col, f * f * c)
		x = x.transpose(2, 1, 0)							# (f*f*c, out_row * out_col, batch)
		x = x.reshape(x.shape[0], -1)						# (f*f*c, out_row * out_col * batch)


		calc_new_n = lambda x: (x - self.ksize) // self.stride + 1
		row, col = matrix.shape[1:3]
		new_shape = (n_outchan, calc_new_n(row), calc_new_n(col), n_batch)

		self.result = w.dot(x).reshape(new_shape)			# (out_chan, out_row, out_col, batch)
		self.result = self.result.transpose(3, 1, 2, 0)		# (batch, out_row, out_col, out_chan)
		return

class BiasAdd(DnnNode):
	def __init__(self, name: str, in_node: DnnNode, biases: np.ndarray):
		if in_node.result.shape[-1] != biases.shape[-1]:
			raise DNNException("input's shape {} != biases.shape {}".format(in_node.result.shape[-1], biases.shape[-1]))
		self.in_node, self.biases = in_node, biases
		self.result = np.ndarray(in_node.result.shape)

		self.name = name
		self._notify_completion(name)
		return

	def run(self):
		matrix = self.in_node.result
		self.result = matrix + self.biases
		return


class MaxPool2D(DnnNode):
	def __init__(self, name, in_node: DnnNode, ksize: list, strides: list, padding: str):
		self._check_quadruple(ksize)
		self._check_quadruple(strides)
		self._check_padding(padding)

		self.in_node = in_node
		self.ksize = ksize[1]
		self.stride = strides[1]
		self.padding = padding
		if(padding.upper() == "SAME"):
			out_w = math.floor(in_node.result.shape[1]/strides[1]) + 1
			out_h = math.floor(in_node.result.shape[2]/strides[2]) + 1
			self.result = np.ndarray((in_node.result.shape[0], out_w, out_h, in_node.result.shape[3]))
		else:
			out_w = math.floor((in_node.result.shape[1] - kernels.shape[0])/strides[1]) + 1
			out_h = math.floor((in_node.result.shape[2] - kernels.shape[1])/strides[2]) + 1
			self.result = np.ndarray((in_node.result.shape[0], out_w, out_h, in_node.result.shape[3]))
						
		self.name = name
		self._notify_completion(name)
		return

		
	def run(self):
		matrix = self.in_node.result
		n_batch, row, col, chan = matrix.shape


		if self.padding.upper() == "SAME":
			x = self._stride(matrix, self.ksize, self.stride, "edge")
		else:
			x = self._stride(matrix, self.ksize, self.stride)
		x = self._make_channel_last(x)		# Result: (batch, f, f, c, out_n * out_n)
											# (f, f) is a tiled local matrix.

		def calc_new_len(n):
			if self.padding.upper() == "SAME":
				n = (n + self._calc_same_pad(n, self.ksize, self.stride))
			return (n - self.ksize) // self.stride + 1

		new_shape = (n_batch, chan, calc_new_len(row), calc_new_len(col))
		# Find a max value among (f, f) 2 by 2 matrix elements
		self.result = x.max(axis=(1, 2))	\
						.reshape(new_shape)	\
						.transpose(0, 2, 3, 1)	# (batch, out_row, out_col, out_chan)
		return

class BatchNorm(DnnNode):
	def __init__(self, name, in_node, mean, variance, gamma, epsilon, beta=0):

		if not (in_node.result.shape[-1] == mean.shape[0]):
			raise DNNException()
		elif not (in_node.result.shape[-1] == variance.shape[0]):
			raise DNNException()
		elif not (in_node.result.shape[-1] == gamma.shape[0]):
			raise DNNException()

		self.in_node = in_node
		self.mean = mean
		self.variance = variance
		self.gamma = gamma
		self.epsilon = epsilon
		self.beta = 0
		self.result = np.ndarray(in_node.result.shape)
		
		self.name = name
		self._notify_completion(name)
		return

	def run(self):
		matrix = self.in_node.result
		x = (matrix - self.mean) / np.sqrt(self.variance + self.epsilon)
		self.result = self.gamma * x + self.beta
		return

class LeakyReLU(DnnNode):
	def __init__(self, name: str, in_node: DnnNode):
		self.in_node = in_node
		self.alpha = 0.1
		self.result = np.ndarray(in_node.result.shape)

		self._notify_completion(name)
		return

	def run(self):
		value = self.in_node.result
		self.result = np.maximum(value, self.alpha * value)



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

