import unittest
import numpy as np

import dnn


class TestNode(unittest.TestCase):
	def test_verify(self):
		lhs = np.array([1, 2, 3])
		rhs = np.array([4, 5])

		function = dnn.DnnNode()._verify_shapes
		self.assertRaises(TypeError, function, lhs, rhs)

	def test_leakyrelu(self):
		arr = np.array([-1, 0, 1, 2, 3])
		lr = dnn.LeakyReLU("leakyReLU", arr)
		lr.run()
		actual = lr.result
		np.testing.assert_array_equal([-0.1, 0, 1, 2, 3], actual)

	def test_biasadd(self):
		lhs = np.array([1, 2, 3, 4, 5])
		rhs = np.array([1, 1, 1, 1, 1])
		ba = dnn.BiasAdd("BiasAdd", lhs, rhs)
		ba.run()
		result = ba.result

		np.testing.assert_array_equal([2, 3, 4, 5, 6], result)


if __name__ == "__main__":
	unittest.main()
