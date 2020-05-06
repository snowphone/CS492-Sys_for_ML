import unittest
import numpy as np

import dnn


class TestNode(unittest.TestCase):
	def test_verify(self):
		lhs = np.array([1, 2, 3])
		rhs = np.array([4, 5])

		function = dnn.DnnNode()._verify_shapes
		self.assertRaises(dnn.DNNException, function, lhs, rhs)

	def test_leakyrelu(self):
		arr = np.array([-1, 0, 1, 2, 3])
		node = dnn.DnnNode()
		node.result = arr
		lr = dnn.LeakyReLU("leakyReLU", node)
		lr.run()
		actual = lr.result
		np.testing.assert_array_equal([-0.1, 0, 1, 2, 3], actual)

	def test_channel_first(self):
		inp = np.array([[1,2], [3,4]])
		fn = dnn.DnnNode()._make_channel_first
		
		np.testing.assert_array_equal(fn(inp), [[1,3], [2,4]])

	def test_channel_last(self):
		inp = np.array([[1,3], [2,4]])
		fn = dnn.DnnNode()._make_channel_last
		
		np.testing.assert_array_equal(fn(inp), [[1,2], [3,4]])

	def test_biasadd(self):
		def make_node():
			node = dnn.DnnNode()
			node.result = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
			node.result = node._make_channel_last(node.result)
			return node
		def make_biases():
			return np.array([1,2])
		def make_answer():
			expected = np.array([[[2, 3], [4, 5]], [[7, 8], [9, 10]]])
			return node._make_channel_last(expected)

		node = make_node()
		biases = make_biases()
		expected = make_answer()

		ba = dnn.BiasAdd("BiasAdd", node, biases)
		ba.run()
		result = ba.result

		np.testing.assert_array_equal(expected, result)

	def test_pad(self):
		'''Squared matrix'''
		func = dnn.DnnNode()._pad
		matrix = np.arange(15).reshape(5, 3)
		n_f = 3
		n_s = 1
		expected = np.array([
			[0, 0, 0, 0, 0],
			[0, 0, 1, 2, 0],
			[0, 3, 4, 5, 0],
			[0, 6, 7, 8, 0],
			[0, 9, 10, 11, 0],
			[0, 12, 13, 14, 0],
			[0, 0, 0, 0, 0]])
		expected = expected + 0

		actual = func(matrix, n_f, n_s)

		np.testing.assert_array_equal(expected, actual)
		
	def test_pad2(self):
		'''rectangled matrix'''
		func = dnn.DnnNode()._pad
		matrix = np.array([[1, 2, 3]])
		n_f = 2
		n_s = 2
		expected = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 2, 3, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			])
		expected = expected + 0

		actual = func(matrix, n_f, n_s)

		np.testing.assert_array_equal(expected, actual)
	
	def test_stride(self):
		'''
		1 x 1
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.arange(2 * 2).reshape(2, 2)

		stride = 1
		ksize = 1
		expected = np.array([
			[[0]], 
			[[1]], 
			[[2]], 
			[[3]]
			])

		actual = strider(matrix, ksize, stride)
		np.testing.assert_array_equal(expected, actual)
		

	def test_stride2(self):
		'''
		evenly divisible by ksize
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.arange(3*3).reshape(3, 3)

		ksize = 2
		stride = 1
		expected = np.array([
			[[0, 1], [3, 4]], [[1, 2], [4, 5]],
			[[3, 4], [6, 7]], [[4, 5], [7, 8]]
			])

		actual = strider(matrix, ksize, stride)
		np.testing.assert_array_equal(expected, actual)

	def test_stride3(self):
		'''
		In vertical view, not evenly divisible
		'''
		strider = dnn.DnnNode()._stride

		matrix = np.array([
			[20,  200,   -5,   23],
			[-13,  134,  119,  100],
			[120,   32,   49,   25],
			[-120,   12,   9,   23],
			[-57,   84,   19,   17],
			])

		stride = 2
		ksize = 2
		expected = np.array([
			[[20, 200], [-13, 134]], [[-5, 23], [119, 100]],
			[[120, 32], [-120, 12]], [[49, 25], [9, 23]],
			[[-57, 84], [-57, 84]], [[19, 17], [19, 17]],
			])

		actual = strider(matrix, ksize, stride, "edge")
		np.testing.assert_array_equal(expected, actual)

	def test_stride4(self):
		strider = dnn.DnnNode()._stride
		stride = 2
		ksize = 2

		matrix= np.array([
			[20,  200,   -5,   23],
			[-13,  134,  119,  100],
			[120,   32,   49,   25],
			[-120,   12,   9,   23],
			#[-57,   84,   19,   17],
			])
		expected = np.array([
			[[20, 200], [-13, 134]], [[-5, 23], [119, 100]],
			[[120, 32], [-120, 12]], [[49, 25], [9, 23]],
			])

		actual = strider(matrix, ksize, stride)
		np.testing.assert_array_equal(expected, actual)



	
	def test_maxpool(self):
		mat = np.array([
			[20,  200,   -5,   23],
			[-13,  134,  119,  100],
			[120,   32,   49,   25],
			[-120,   12,   9,   23],
			#[-57,   84,   19,   17],
			])
		mat = mat.reshape(*mat.shape, 1)
		expected = np.array([
			[200, 119],
			[120, 49],
			#[84, 19],
			])
		expected = expected.reshape(*expected.shape, 1)
		in_node = dnn.DnnNode()
		in_node.result = mat

		pooler = dnn.MaxPool2D("max_pool2d", in_node, [1,2,2,1], [1,2,2,1], "valid")
		pooler.run()

		actual = pooler.result
		np.testing.assert_array_equal(expected, actual)

	#def test_maxpool2(self):
	#	mat = np.array([
	#		[20,  200,   -5,   23],
	#		[-13,  134,  119,  100],
	#		[120,   32,   49,   25],
	#		[-120,   12,   9,   23],
	#		[-57,   84,   19,   17],
	#		])
	#	mat = mat.reshape(*mat.shape, 1)
	#	expected = np.array([
	#		[200, 119],
	#		[120, 49],
	#		[84, 19],
	#		])
	#	expected = expected.reshape(*expected.shape, 1)
	#	in_node = dnn.DnnNode()
	#	in_node.result = mat

	#	pooler = dnn.MaxPool2D("max_pool2d", in_node, [1,2,2,1], [1,2,2,1], "valid")
	#	pooler.run()

	#	actual = pooler.result
	#	np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
	unittest.main()
