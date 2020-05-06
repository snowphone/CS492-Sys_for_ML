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

	def _verify_shapes(self, lhs: np.ndarray, rhs: np.ndarray):
		'''
		Compare the shapes of given numpy arrays.
		If the shape is not equal, then DNNException would be raised to interrupt the flow.

		@throws TypeError
		'''
		if lhs.shape != rhs.shape:
			className = self.__class__.__name__
			raise DNNException("While constructing {}, lhs's shape {} != rhs's shape {}".format(className, lhs.shape, rhs.shape))

		return

	def _make_channel_last(self, chan_first: np.ndarray, has_batch=True)->np.ndarray:
		chan_first = np.array(chan_first)
		last_dim = len(chan_first.shape) - 1
		first_dim = 1 if has_batch else 0
		return np.moveaxis(chan_first, first_dim, last_dim) # Or, you can implement using np.transpose

	def _make_channel_first(self, chan_last: np.ndarray, has_batch=True) -> np.ndarray:
		chan_last = np.array(chan_last)
		last_dim = len(chan_last.shape) - 1
		first_dim = 1 if has_batch else 0
		return np.moveaxis(chan_last, last_dim, first_dim)

	def _stride(self, matrix: np.ndarray, ksize: int, stride: int, pad_mode="constant")->np.ndarray:
		'''
		TODO: Parallelize!

		@return np.ndarray shape: (batch, out_n x out_n, tile)  => (batch, n*n, f, f, c)
		Tile means a local area of the matrix and its shape equals to the kernel.
		'''
		#strider = np.lib.stride_tricks.as_strided


		row, col = matrix.shape[1:3]
		calc_pad = lambda n: max(ksize - stride, 0) \
				if n % stride == 0 \
				else max(ksize - (n % stride), 0)
		r_pad = calc_pad(row)
		c_pad = calc_pad(col)
		if pad_mode == "edge":
			pad = [(0, r_pad), (0, c_pad)]
			matrix = self._pad(matrix, ksize, stride, pad_mode, pad=pad)

		matrix = self._make_channel_last(matrix, has_batch=False)


		
		formula = lambda x: (x - ksize) // stride + 1
		padded_row, padded_col = matrix.shape[:2]


		tiles = np.array([
			matrix[r:r+ksize, c:c+ksize]
			for r in range(0, row, stride) 
			for c in range(0, col, stride)
			if r != row and c != col and 
				r + ksize <=padded_row and c + ksize <= padded_col
			])

		tiles = self._make_channel_first(tiles, has_batch=False)

		return tiles

	def _pad(self, matrix: np.ndarray, ksize: int, 
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
		@params pad [(v_pad, v_pad), (h_pad, h_pad)] 
				If pad is set, then ksize and n_stride are ignored.
		'''
		matrix = self._make_channel_last(matrix, has_batch=False)
		if pad is None:
			n_row, n_col = matrix.shape[:2]
			v_pad = ((n_stride - 1) * n_row - 1 + ksize) // 2
			h_pad = ((n_stride - 1) * n_col - 1 + ksize) // 2

			pad = [(v_pad, v_pad), (h_pad, h_pad)]	# [(up, down), (left, right)]

		pad = [*pad, (0, 0), (0, 0)]	# First zeroes are for channels, seconds are for batches

		matrix = np.pad(matrix, pad_width=pad, mode=mode)

		matrix = self._make_channel_first(matrix, has_batch=False)


		return matrix

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



#
# Complete below classes.
#

class Conv2D(DnnNode):
	def __init__(self, name: str, in_node: DnnNode, 
			kernels: np.ndarray, strides: list, padding: str):
		'''

		@param ("SAME" | "VALID") padding
		'''
		# Need some verification codes...
		#if in_node.result.shape[-1] != kernels.shape[-2]:
		#	raise DNNException("the number of output channels is different")
		self._check_quadruple(strides)
		self._check_padding(padding)

		self.ksize = kernels.shape[0]
		self.stride = strides[1]
		self.in_node = in_node
		self.kernels = kernels
		self.padding = padding

		self.name = name
		self._notify_completion(name)

	def run(self):
		# Strategy 1: _stride => dot product => reshape
		# Strategy 2: (outchan x kernel) @ (tile x strides).
		#			each is a matrix and since kernel.length == tile.length matmul can be applied!

		# Strategy 2:
		#print(" {}.run ".format(self.name).center(30, '@'))
		#print(f"ksize: {self.ksize}, stride: {self.stride}")
		matrix = self.in_node.result

		#print(matrix.shape)
		if self.padding.upper() == "SAME":
			matrix = self._pad(matrix, self.ksize, self.stride)
		#print(matrix.shape)


		#print(" weight ".center(30, '@'))
		n_outchan = self.kernels.shape[-1]
		w = self._make_channel_first(self.kernels, has_batch=False)
		#print(w.shape)
		w = w.reshape(n_outchan, -1)	# (out_chan, f * f * c)
		#print(w.shape)


		#print(" input ".center(30, '@'))
		n_batch = matrix.shape[0]
		#print(matrix.shape)
		tmp_x = self._stride(matrix, self.ksize, self.stride)
		#print(tmp_x.shape)
		tmp_x = tmp_x.reshape(n_batch, tmp_x.shape[1], -1)	# (n_batch, out_n * out_n, f*f*c)
		#print(tmp_x.shape)
		tmp_x = tmp_x.reshape(n_batch, tmp_x.shape[1], -1)	# (n_batch, out_n * out_n, f*f*c)
		#print(tmp_x.shape)
		x = tmp_x.transpose(2, 1, 0)		# (f*f*c, out_n * out_n, batch)
		#print(x.shape)
		x = x.reshape(x.shape[0], -1)		# (f*f*c, out_n * out_n * batch)
		#print(x.shape)


		formula = lambda x: (x - self.ksize) // self.stride + 1
		row, col = matrix.shape[1:3]
		#print(f"row: {row}, col: {col}")
		new_shape = (n_outchan, formula(row), formula(col), n_batch)

		self.result = w.dot(x).reshape(new_shape)	# (out_chan, out_n, out_n * batch)
		self.result = self.result.transpose(3, 1, 2, 0)
		return

class BiasAdd(DnnNode):
	def __init__(self, name: str, in_node: DnnNode, biases: np.ndarray):
		#self._verify_shapes(in_node.result, biases, dim=-1)
		#if in_node.result.shape[-1] != biases.shape[-1]:
		#	raise DNNException("input's shape {} != biases.shape {}".format(in_node.result.shape[-1], biases.shape[-1]))
		self.in_node, self.biases = in_node, biases
		self.result = None

		self._notify_completion(name)
		return

	def run(self):
		matrix = self.in_node.result
		#node = self._make_channel_first(node)
		#node = [n + bias for n, bias in zip(node, self.biases)]
		#self.result = self._make_channel_last(node)
		self.result = matrix + self.biases	# Same as the above theee lines, but it is much faster due to parallelism.
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


		self._notify_completion(name)
		return

		
	def run(self):
		'''
		valid 모드에서는 잘 동작하므로, 이에 관계된 코드는 지우지 말 것!
		'''
		matrix = self.in_node.result
		n_batch, row, col, *depths = matrix.shape

		print("#" * 40)
		print(f"k: {self.ksize}, s: {self.stride}")


		print(f"matrix: {matrix.shape}")
		#if self.padding.upper() == "SAME":
		#	calc_pad = lambda n: max(self.ksize - self.stride, 0) \
		#			if n % self.stride == 0 \
		#			else max(self.ksize - (n % self.stride), 0)
		#	r_pad = calc_pad(row)
		#	c_pad = calc_pad(col)
		#	print("rpad", r_pad, "cpad", c_pad)
		#	pad = [(0, r_pad), (0, c_pad)]


		print(f"matrix: {matrix.shape}")

		def formula(n):
			if self.padding.upper() == "SAME":
				return int(np.ceil(float(n - self.ksize) / self.stride) + 1)
			else:
				return (n - self.ksize) // self.stride + 1

		if self.padding.upper() == "SAME":
			tmp_x = self._stride(matrix, self.ksize, self.stride, "edge")
		else:
			tmp_x = self._stride(matrix, self.ksize, self.stride)
		x = self._make_channel_last(tmp_x)	# Result: (batch, f, f, c, out_n * out_n)
											# (f, f) is the tiled local matrix.
		print(f"x: {x.shape}")
		self.result = x.max(axis=(1, 2))	# Find a max value among (f, f) matrix elements
		new_shape = (n_batch, *depths, formula(row), formula(col))
		print(f"result: {self.result.shape}")
		print(f"new shape: {new_shape}")
		self.result = self.result.reshape(new_shape).transpose(0, 2, 3, 1)
		return

class BatchNorm(DnnNode):
	def __init__(self, name, in_node, mean, variance, gamma, epsilon, beta=0):

		#if not (in_node.result.shape[-1] == mean.shape[0]):
		#	raise DNNException()
		#elif not (in_node.result.shape[-1] == variance.shape[0]):
		#	raise DNNException()
		#elif not (in_node.result.shape[-1] == gamma.shape[0]):
		#	raise DNNException()

		self.in_node = in_node
		self.mean = mean
		self.variance = variance
		self.gamma = gamma
		self.epsilon = epsilon
		self.beta = 0

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

		self._notify_completion(name)
		return

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

