import os
import sys
from datetime import datetime
from functools import reduce, wraps
from typing import List, Tuple

import cv2
import numpy as np

import yolov2tiny


def measure(func):
	""" Measure how long a function takes time """
	@wraps(func)
	def impl(*args, **kargs):
		beg=datetime.now()
		ret = func(*args, **kargs)
		time = (datetime.now() - beg).total_seconds()
		print("{}: {}s".format(func.__name__, time))
		return ret

	return impl


def open_video_with_opencv(
        in_video_path: str,
        out_video_path: str) -> (cv2.VideoCapture, cv2.VideoWriter):

	reader = cv2.VideoCapture(in_video_path)
	if not reader.isOpened():
		raise Exception("Failed to open \'{}\'".format(in_video_path))

	fps = reader.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

	writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
	if not writer.isOpened():
		raise Exception(
		    "Failed to create video named \'{}\'".format(out_video_path))

	return reader, writer


def resize_input(im: np.ndarray) -> np.ndarray:
	imsz = cv2.resize(im, (416, 416), interpolation=cv2.INTER_AREA)
	imsz = imsz / 255.
	imsz = imsz[:, :, ::-1]
	imsz = np.asarray(imsz, dtype=np.float32)
	return imsz.reshape((1, *imsz.shape))


color_t = Tuple[float, float, float]
coord_t = Tuple[int, int]
proposal_t = Tuple[str, coord_t, coord_t, color_t]


def restore_shape(proposals: List[proposal_t], restore_width: int,
                  restore_height: int) -> List[proposal_t]:
	"""
	Read proposal list and reshape proposal coordinates into original video's resolution
	"""
	def reshape(record: proposal_t) -> proposal_t:
		"""
		Get a record and reshape coordinates into original ratio.
		cf) lu means left upper and rb means right bottom.
		"""
		calc_coord = lambda x, new_d: np.clip(int(x / 416 * new_d), 0, new_d)
		name, (lux, luy), (rbx, rby), color = record
		lux, rbx = map(lambda x: calc_coord(x, restore_width), [lux, rbx])
		luy, rby = map(lambda y: calc_coord(y, restore_height), [luy, rby])
		return (name, (lux, luy), (rbx, rby), color)

	return [reshape(it) for it in proposals]


def draw(image: np.ndarray, proposals: List[proposal_t]) -> np.ndarray:
	'''
	Draw bounding boxes into image and return it

	proposals contains a list of (best_class_name, lefttop, rightbottom, color).
	'''
	for name, lefttop, rightbottom, color in proposals:
		height, width, _channel = image.shape

		cv2.rectangle(image, lefttop, rightbottom, color, 2)
		cv2.putText(image, name, (lefttop[0], max(0, lefttop[1] - 10)),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return image


def store_tensors(tensors: List[np.ndarray]):
	os.makedirs("intermediate", exist_ok=True)
	for i, tensor in enumerate(tensors):
		path = os.path.join("intermediate", "layer_{}.npy".format(i))
		np.save(path, tensor)


@measure
def video_object_detection(in_video_path: str,
                           out_video_path: str,
                           proc="cpu"):
	"""
	Read a videofile, scan each frame and draw objects using pretrained yolo_v2_tiny model.
	Finally, store drawed frames into 'out_video_path'
	"""
	reader, writer = open_video_with_opencv(in_video_path, out_video_path)
	yolo = yolov2tiny.YOLO_V2_TINY((416, 416, 3), "./y2t_weights.pickle", proc)

	width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

	acc, firstTime = [], True
	while reader.isOpened():
		okay, original_image = reader.read()
		if not okay:
			break
		beg_start = datetime.now()
		image = resize_input(original_image)
		beg_infer = datetime.now()
		batched_tensors_list = yolo.inference(image)
		inference_time = (datetime.now() - beg_infer).total_seconds()

		if firstTime:
			store_tensors(map(lambda x: x[0], batched_tensors_list))	# Remove batch shape
			firstTime = False

		tensor = batched_tensors_list[-1][0]

		proposals = yolov2tiny.postprocessing(tensor)
		proposals = restore_shape(proposals, width, height)
		out_image = draw(original_image, proposals)
		writer.write(out_image)

		end_to_end_time = (datetime.now() - beg_start).total_seconds()
		acc.append((inference_time, end_to_end_time))
		print("#{} inference: {:.3f}\tend-to-end: {:.3f}".format(len(acc), inference_time, end_to_end_time))

	reader.release()
	writer.release()
	inference_sum, end_to_end_sum = reduce(lambda x,y: (x[0] + y[0], x[1] + y[1]), acc)
	size = len(acc)
	print("Average inference: {:.3f}s\taverage end-to-end: {:.3f}s".format(inference_sum/size, end_to_end_sum/size))
	print("Throughput: {:.3f}fps".format(size / end_to_end_sum))
	return


def main():
	if len(sys.argv) < 3:
		print(
		    "Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])"
		)
		sys.exit()

	in_video_path = sys.argv[1]
	out_video_path = sys.argv[2]

	if len(sys.argv) == 4:
		proc = sys.argv[3]
	else:
		proc = "cpu"

	video_object_detection(in_video_path, out_video_path, proc)


if __name__ == "__main__":
	main()
