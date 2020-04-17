import os
import pickle
import sys
from typing import List

import numpy as np
import tensorflow as tf

class YOLO_V2_TINY(object):
	def __init__(self, in_shape, weight_pickle, proc="cpu"):
		self.g = tf.Graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.7
		self.sess = tf.Session(config=config, graph=self.g)
		self.proc = proc
		self.weight_pickle = weight_pickle
		self.batchSize = 1
		self.input_tensor, self.layers = self.build_graph(in_shape)

	def build_graph(self, in_shape):

		# Auxiliary functions
		def conv_b(in_tensor, weights) -> List[tf.Tensor]:
			"""
			It's a composite function of conv |> bias.
			returns a list of layers
			"""
			l1 = conv(in_tensor, weights["kernel"])
			l2 = bias(l1, weights["biases"])
			return [l1, l2]

		def conv_b_batNorm_lRelu(in_tensor, weights) -> List[tf.Tensor]:
			"""
			It's a composite function of conv |> bias |> batch normalization |> leakyRelu.
			returns a list of layers
			"""
			l_list = conv_b(in_tensor, weights)
			l2 = batch_norm(l_list[-1], weights)
			l3 = leakyRelu(l2)
			return [*l_list, l2, l3]

		def conv_b_batNorm_lRelu_maxpool(in_tensor,
		                                 weights,
		                                 pool_stride) -> List[tf.Tensor]:
			"""
			It's a composite function of conv |> bias |> batch normalization |> leakyRelu |> maxpool.
			returns a list of layers
			"""
			l_list = conv_b_batNorm_lRelu(in_tensor, weights)
			l2 = maxpool(l_list[-1], pool_stride)
			return [*l_list, l2]

		# layers is a 2d array which contains a list of sublayers
		# For example, layers looks like [[conv0, bias0, bat_norm0], [conv1, ...], ...]
		layers = []
		with open(self.weight_pickle, "rb") as f:
			weights_list = pickle.load(
			    f, encoding="latin1")  # type: List[Dict[str, np.ndarray]]

		with self.g.as_default(), self.g.device(self.proc):
			x = tf.placeholder(tf.float32, (self.batchSize, *in_shape), name="input")
			l1 = conv_b_batNorm_lRelu_maxpool(x, weights_list[0], [2, 2]);			layers.append(l1)
			l2 = conv_b_batNorm_lRelu_maxpool(l1[-1], weights_list[1], [2, 2]);		layers.append(l2)
			l3 = conv_b_batNorm_lRelu_maxpool(l2[-1], weights_list[2], [2, 2]);		layers.append(l3)
			l4 = conv_b_batNorm_lRelu_maxpool(l3[-1], weights_list[3], [2, 2]);		layers.append(l4)
			l5 = conv_b_batNorm_lRelu_maxpool(l4[-1], weights_list[4], [2, 2]);		layers.append(l5)
			l6 = conv_b_batNorm_lRelu_maxpool(l5[-1], weights_list[5], [1, 1]);		layers.append(l6)
			l7 = conv_b_batNorm_lRelu(l6[-1], weights_list[6]);				layers.append(l7)
			l8 = conv_b_batNorm_lRelu(l7[-1], weights_list[7]);				layers.append(l8)
			l9 = conv_b(l8[-1], weights_list[8]);						layers.append(l9)

		with self.g.as_default(), self.g.device(self.proc):
			self.sess.run(tf.global_variables_initializer())


		def flatten(table: List[List[tf.Tensor]]) -> List[tf.Tensor]:
			""" Flatten 2-d array into 1-d array """
			return sum(table, [])

		return x, flatten(layers)

	def inference(self, img):
		feed_dict = {self.input_tensor: img}
		out_tensors = self.sess.run(self.layers, feed_dict)
		return out_tensors


def postprocessing(predictions):

	n_classes = 20
	n_grid_cells = 13
	n_b_boxes = 5
	n_b_box_coord = 4

	# Names and colors for each class
	classes = [
	    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
	    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
	]
	colors = [(254.0, 254.0, 254),
	          (239.88888888888889, 211.66666666666669, 127),
	          (225.77777777777777, 169.33333333333334, 0),
	          (211.66666666666669, 127.0, 254),
	          (197.55555555555557, 84.66666666666667, 127),
	          (183.44444444444443, 42.33333333333332, 0),
	          (169.33333333333334, 0.0, 254),
	          (155.22222222222223, -42.33333333333335, 127),
	          (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
	          (112.88888888888889, 211.66666666666669, 127),
	          (98.77777777777777, 169.33333333333334, 0),
	          (84.66666666666667, 127.0, 254),
	          (70.55555555555556, 84.66666666666667, 127),
	          (56.44444444444444, 42.33333333333332, 0),
	          (42.33333333333332, 0.0, 254),
	          (28.222222222222236, -42.33333333333335, 127),
	          (14.111111111111118, -84.66666666666664, 0), (0.0, 254.0, 254),
	          (-14.111111111111118, 211.66666666666669, 127)]

	# Pre-computed YOLOv2 shapes of the k=5 B-Boxes
	anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

	thresholded_predictions = []

	# IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
	predictions = predictions.reshape((13, 13, 5, 25))

	# IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
	for row in range(n_grid_cells):
		for col in range(n_grid_cells):
			for b in range(n_b_boxes):

				tx, ty, tw, th, tc = predictions[row, col, b, :5]

				# IMPORTANT: (416 img size) / (13 grid cells) = 32!
				# YOLOv2 predicts parametrized coordinates that must be converted to full size
				# final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
				center_x = (float(col) + sigmoid(tx)) * 32.0
				center_y = (float(row) + sigmoid(ty)) * 32.0

				roi_w = np.exp(tw) * anchors[2 * b + 0] * 32.0
				roi_h = np.exp(th) * anchors[2 * b + 1] * 32.0

				final_confidence = sigmoid(tc)

				# Find best class
				class_predictions = predictions[row, col, b, 5:]
				class_predictions = softmax(class_predictions)

				class_predictions = tuple(class_predictions)
				best_class = class_predictions.index(max(class_predictions))
				best_class_score = class_predictions[best_class]

				# Flip the coordinates on both axes
				left = int(center_x - (roi_w / 2.))
				right = int(center_x + (roi_w / 2.))
				top = int(center_y - (roi_h / 2.))
				bottom = int(center_y + (roi_h / 2.))

				if ((final_confidence * best_class_score) > 0.3):
					thresholded_predictions.append(
					    [[left, top, right,
					      bottom], final_confidence * best_class_score,
					     classes[best_class]])

	# Sort the B-boxes by their final score
	thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)

	# Non maximal suppression
	if (len(thresholded_predictions) > 0):
		nms_predictions = non_maximal_suppression(thresholded_predictions, 0.3)
	else:
		nms_predictions = []

	label_boxes = []
	# Append B-Boxes
	for i in range(len(nms_predictions)):

		best_class_name = nms_predictions[i][2]
		lefttop = tuple(nms_predictions[i][0][0:2])
		rightbottom = tuple(nms_predictions[i][0][2:4])
		color = colors[classes.index(nms_predictions[i][2])]

		label_boxes.append((best_class_name, lefttop, rightbottom, color))

	return label_boxes


def iou(boxA, boxB):
	# boxA = boxB = [x1,y1,x2,y2]

	# Determine the coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# Compute the area of intersection
	intersection_area = (xB - xA + 1) * (yB - yA + 1)

	# Compute the area of both rectangles
	boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# Compute the IOU
	iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

	return iou


def non_maximal_suppression(thresholded_predictions, iou_threshold):

	nms_predictions = []

	# Add the best B-Box because it will never be deleted
	nms_predictions.append(thresholded_predictions[0])

	# For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
	# thresholded_predictions[i][0] = [x1,y1,x2,y2]
	i = 1
	while i < len(thresholded_predictions):
		n_boxes_to_check = len(nms_predictions)
		to_delete = False

		j = 0
		while j < n_boxes_to_check:
			curr_iou = iou(thresholded_predictions[i][0],
			               nms_predictions[j][0])
			if (curr_iou > iou_threshold):
				to_delete = True
			j = j + 1

		if to_delete == False:
			nms_predictions.append(thresholded_predictions[i])
		i = i + 1

	return nms_predictions


def sigmoid(x):
	return 1 / (1 + np.e**-x)


def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)


def conv(in_tensor, weight) -> tf.Tensor:
	filter = tf.Variable(weight)
	return tf.nn.conv2d(in_tensor, filter, [1, 1, 1, 1], "SAME")


def batch_norm(in_tensor, weights):
	mean, variance, gamma, beta = weights["moving_mean"], weights[
	    "moving_variance"], weights["gamma"], None
	return tf.nn.batch_normalization(in_tensor, mean, variance, beta, gamma, 1e-5)


def leakyRelu(x):
	return tf.maximum(x, 0.1 * x)


def bias(in_tensor, weight: np.ndarray) -> tf.Tensor:
	bias = tf.Variable(weight)
	return tf.nn.bias_add(in_tensor, bias)


def maxpool(x, stride):
	return tf.nn.max_pool2d(x, [2,2], stride, "SAME")
