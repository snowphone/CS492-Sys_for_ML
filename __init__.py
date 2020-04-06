import sys
import os
import numpy as np
import cv2
import datetime
import yolov2tiny
from typing import List, Tuple


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

	time_acc, frames_acc, firstTime = 0, 0, True
	while reader.isOpened():
		okay, original_image = reader.read()
		if not okay:
			break

		image = resize_input(original_image)
		beg = datetime.datetime.now()
		batched_tensors_list = yolo.inference(image)
		end = datetime.datetime.now()
		if firstTime:
			store_tensors(batched_tensors_list)
			firstTime = False

		batched_tensors = batched_tensors_list[-1]

		inference_time = (end - beg).total_seconds()

		time_acc += inference_time
		frames_acc += 1

		print(
		    "#{}: done in {:.3f} seconds\ttotal:{:.3f} seconds\tthroughput: {:.3f} frames per second"
		    .format(frames_acc, inference_time, time_acc,
		            frames_acc / time_acc))

		for tensor in batched_tensors:
			proposals = yolov2tiny.postprocessing(tensor)
			proposals = restore_shape(proposals, width, height)
			out_image = draw(original_image, proposals)
			writer.write(out_image)

	reader.release()
	writer.release()
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

