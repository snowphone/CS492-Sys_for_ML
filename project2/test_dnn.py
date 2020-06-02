import unittest
import numpy as np

import dnn


class TestNode(unittest.TestCase):
	def test_leakyrelu(self):
		arr = np.array([-1, 0, 1, 2, 3])
		node = dnn.DnnNode()
		node.result = arr
		lr = dnn.LeakyReLU("leakyReLU", node)
		lr.run()
		actual = lr.result
		np.testing.assert_array_equal([-0.1, 0, 1, 2, 3], actual)

	def test_channel_first(self):
		matrix = np.array([[1,2], [3,4]])[np.newaxis, :]
		fn = dnn.DnnNode()._make_channel_first
		expected = np.array([[1,3], [2,4]])[np.newaxis, :]
		actual = fn(matrix)
		
		np.testing.assert_array_equal(expected, actual)

	def test_channel_last(self):
		matrix = np.array([[1,3], [2,4]])[np.newaxis, :]
		fn = dnn.DnnNode()._make_channel_last
		expected = np.array([[1,2], [3,4]])[np.newaxis, :]
		actual = fn(matrix)
		
		np.testing.assert_array_equal(expected, actual)

	def test_biasadd(self):
		def make_node():
			node = dnn.DnnNode()
			node.result = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])[np.newaxis, :]
			node.result = node._make_channel_last(node.result)
			return node
		def make_biases():
			return np.array([1,2])[np.newaxis, :]
		def make_answer():
			expected = np.array([[[2, 3], [4, 5]], [[7, 8], [9, 10]]])[np.newaxis, :]
			return node._make_channel_last(expected)

		node = make_node()
		biases = make_biases()
		expected = make_answer()

		ba = dnn.BiasAdd("BiasAdd", node, biases)
		ba.run()
		result = ba.result

		np.testing.assert_array_equal(expected, result)


	def test_pad_square(self):
		'''Squared matrix'''
		func = dnn.DnnNode()._pad
		matrix = np.arange(15).reshape(5, 3)[np.newaxis, :, :, np.newaxis]
		n_f = 3
		n_s = 1
		expected = np.array([
			[0, 0, 0, 0, 0],
			[0, 0, 1, 2, 0],
			[0, 3, 4, 5, 0],
			[0, 6, 7, 8, 0],
			[0, 9, 10, 11, 0],
			[0, 12, 13, 14, 0],
			[0, 0, 0, 0, 0]])[np.newaxis, :, :, np.newaxis]

		actual = func(matrix, n_f, n_s)

		np.testing.assert_array_equal(expected, actual)
		
	def test_pad_rectangle(self):
		'''rectangled matrix'''
		func = dnn.DnnNode()._pad
		matrix = np.array([[1, 2, 3]])[np.newaxis, :, :, np.newaxis]
		n_f = 2
		n_s = 2
		expected = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 2, 3, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			])[np.newaxis, :, :, np.newaxis]

		actual = func(matrix, n_f, n_s)

		np.testing.assert_array_equal(expected, actual)
	
	def test_stride_basic(self):
		'''
		1 x 1
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.arange(2 * 2).reshape(2, 2)[np.newaxis, :, :, np.newaxis]

		stride = 1
		ksize = 1
		expected = np.array([
			[[0, ], ], 
			[[1, ], ], 
			[[2, ], ], 
			[[3, ], ],
			])[np.newaxis, :, :, np.newaxis]

		actual = strider(matrix, ksize, stride)
		np.testing.assert_array_equal(expected, actual)
		

	def test_stride_even(self):
		'''
		evenly divisible by ksize
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.arange(3*3).reshape(3, 3)[np.newaxis, :, :, np.newaxis]

		ksize = 2
		stride = 1
		expected = np.array([
			[[0, 1], [3, 4]], [[1, 2], [4, 5]],
			[[3, 4], [6, 7]], [[4, 5], [7, 8]]
			])[np.newaxis, :, :, :, np.newaxis]

		actual = strider(matrix, ksize, stride)
		np.testing.assert_array_equal(expected, actual)

	def test_stride_vertical_not_even(self):
		'''
		In vertical order, not evenly divisible
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.array([
			[20,  200,   -5,   23],
			[-13,  134,  119,  100],
			[120,   32,   49,   25],
			[-120,   12,   9,   23],
			[-57,   84,   19,   17],
			])[np.newaxis, :, :, np.newaxis]

		stride = 2
		ksize = 2
		expected = np.array([
			[[20, 200], [-13, 134]], [[-5, 23], [119, 100]],
			[[120, 32], [-120, 12]], [[49, 25], [9, 23]],
			[[-57, 84], [-57, 84]], [[19, 17], [19, 17]],
			])[np.newaxis, :, :, :, np.newaxis]

		actual = strider(matrix, ksize, stride, "edge")
		np.testing.assert_array_equal(expected, actual)


	def test_stride_all_not_even(self):
		'''
		In vertical and horizontal order, not evenly divisible
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.array([
			[20,  200,   -5,   23, 7],
			[-13,  134,  119,  100, 8],
			[120,   32,   49,   25, 12],
			[-120,   12,   9,   23, 15],
			[-57,   84,   19,   17, 82],
			])[np.newaxis, :, :, np.newaxis]

		stride = 2
		ksize = 2
		expected = np.array([
			[[20, 200], [-13, 134]], [[-5, 23], [119, 100]], [[7, 7], [8, 8]],
			[[120, 32], [-120, 12]], [[49, 25], [9, 23]], [[12, 12], [15, 15]],
			[[-57, 84], [-57, 84]], [[19, 17], [19, 17]], [[82, 82], [82, 82]],
			])[np.newaxis, :, :, :, np.newaxis]

		actual = strider(matrix, ksize, stride, "edge")
		np.testing.assert_array_equal(expected, actual)


	def test_conv(self):
		mat = np.array([
			[1, 1, 1, 0, 0],
			[0, 1, 1, 1, 0], 
			[0, 0, 1, 1, 1],
			[0, 0, 1, 1, 0],
			[0, 1, 1, 0, 0],
			])[np.newaxis, :, :, np.newaxis]

		kernels = np.array([
			[1, 0, 1],
			[0, 1, 0],
			[1, 0, 1],
			])[:, :, np.newaxis, np.newaxis]

		expected = np.array([
			[4,3,4],
			[2,4,3],
			[2,3,4],
			])[np.newaxis, :, :, np.newaxis]

		in_node = dnn.DnnNode()
		in_node.result = mat
		conv = dnn.Conv2D("conv2d", in_node, kernels, [1,1,1,1], "valid")
		conv.run()

		actual = conv.result
		np.testing.assert_array_equal(expected, actual)

	
	def test_maxpool_valid(self):
		mat = np.array([
			[20,  200,   -5,   23],
			[-13,  134,  119,  100],
			[120,   32,   49,   25],
			[-120,   12,   9,   23],
			])[np.newaxis, :, :, np.newaxis]
		expected = np.array([
			[200, 119],
			[120, 49],
			])[np.newaxis, :, :, np.newaxis]
		in_node = dnn.DnnNode()
		in_node.result = mat

		pooler = dnn.MaxPool2D("max_pool2d", in_node, [1,2,2,1], [1,2,2,1], "valid")
		pooler.run()

		actual = pooler.result
		np.testing.assert_array_equal(expected, actual)

	def test_maxpool_same(self):
		mat = np.array([
			[20,  200,   -5,   23, 1],
			[-13,  134,  119,  100, -1],
			[120,   32,   49,   25, 28],
			[-120,   12,   9,   23, -123],
			])[np.newaxis, :, :, np.newaxis]
		expected = np.array([
			[200, 119, 1],
			[120, 49, 28],
			])[np.newaxis, :, :, np.newaxis]
		in_node = dnn.DnnNode()
		in_node.result = mat

		pooler = dnn.MaxPool2D("max_pool2d", in_node, [1,2,2,1], [1,2,2,1], "same")
		pooler.run()

		actual = pooler.result
		np.testing.assert_array_equal(expected, actual)


	
if __name__ == "__main__":
	unittest.main()
