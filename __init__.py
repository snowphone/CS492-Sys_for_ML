import sys
import os
import numpy as np
import cv2
import datetime
import yolov2tiny
from typing import List, Tuple
from debug import trace

DEBUG=print		# trace debug information

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
def restore_shape(proposals: List[proposal_t], restore_width: int, restore_height: int) -> List[proposal_t]:
	"""
	Read proposal list and reshape proposal coordinates into original video's resolution
	"""
	def reshape(record: proposal_t)  -> proposal_t:
		"""
		Get a record and reshape coordinates into original ratio.
		cf) lu means left upper and rb means right bottom.
		"""
		name, (lux, luy), (rbx, rby), color = record
		DEBUG(lux, luy, rbx, rby, end='\t')
		lux, rbx = map(lambda x: int(x / 416 * restore_width), [lux, rbx])
		luy, rby = map(lambda y: int(y / 416 * restore_height), [luy, rby])
		DEBUG("->", lux, luy, rbx, rby)
		return (name, (lux, luy), (rbx, rby), color)

	return [reshape(it) for it in proposals]

@trace
def draw(image: np.ndarray, proposals: List[proposal_t]) -> np.ndarray:
	'''
	Draw bounding boxes into image and return it

	proposals contains a list of (best_class_name, lefttop, rightbottom, color).
	'''
	if len(image.shape) == 4:
		image = image.reshape(*image.shape[1:])

	DEBUG("image: {}".format(image.shape))
	DEBUG(proposals[0])

	for name, lefttop, rightbottom, color in proposals:
		height, width, _channel = image.shape
		lefttop = int(lefttop[0] * width), int(lefttop[1] * height)
		rightbottom = int(lefttop[0] * width), int(lefttop[1] * height)

		image = cv2.rectangle(image, lefttop, rightbottom, color, 3)
		cv2.putText(image, name, (lefttop[0], lefttop[1] - 10),
		            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)

	return image


def store_tensors(tensors: List[np.ndarray]) -> None:
	os.makedirs("intermediate", exist_ok=True)
	for i, tensor in enumerate(tensors):
		path = os.path.join("intermediate", "layer_{}.npy".format(i))
		np.save(path, tensor)

@trace
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

	acc_time, acc_cnt, firstTime = 0, 0, True
	while reader.isOpened():
		okay, image = reader.read()
		if not okay:
			raise Exception(
			    "An error occured while reading from {}".format(in_video_path))

		image = resize_input(image)
		beg = datetime.datetime.now()
		batched_tensors_list = yolo.inference(image)
		if firstTime:
			store_tensors(batched_tensors_list)
			firstTime = False

		batched_tensors = batched_tensors_list[-1]

		DEBUG(f"batched tensor: {batched_tensors.shape}")
		end = datetime.datetime.now()
		inference_time = (end - beg).total_seconds()

		acc_time += inference_time
		acc_cnt += 1

		print(
				"#{}: done in {:.3f} seconds\ttotal:{:.3f} seconds\tthroughput: {:.3f} frames per second"
		    .format(acc_cnt, inference_time, acc_time, acc_cnt / acc_time))

		for tensor in batched_tensors:
			proposals = yolov2tiny.postprocessing(tensor)
			proposals = restore_shape(proposals, width, height)
			out_image = draw(image, proposals)
			writer.write(out_image)

	reader.release()
	writer.release()

	return

	# 4. Save the intermediate values for the first layer.


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
