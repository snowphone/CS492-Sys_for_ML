import sys
import numpy as np
import cv2
import datetime
import yolov2tiny


def open_video_with_opencv(
        in_video_path: str,
        out_video_path: str) -> (cv2.VideoCapture, cv2.VideoWriter):

	reader = cv2.VideoCapture(in_video_path)
	if not reader.isOpened():
		raise Exception("Failed to open \'{}\'".format(in_video_path))

	fps = reader.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
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


def draw(image: np.ndarray, proposals: list) -> np.ndarray:
	'''
	Draw bounding boxes into image and return it

	proposals contains a list of (best_class_name, lefttop, rightbottom, color).
	'''
	for name, lefttop, rightbottom, color in proposals:
		height, width, _channel = image.shape
		lefttop = int(lefttop[0] * width), int(lefttop[1] * height)
		rightbottom = int(lefttop[0] * width), int(lefttop[1] * height)

		image = cv2.rectangle(image, lefttop, rightbottom, color, 3)
		cv2.putText(image, name, (lefttop[0], lefttop[1] - 10),
		            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)

	return image


def video_object_detection(in_video_path: str,
                           out_video_path: str,
                           proc="cpu"):
	"""
	Read a videofile, scan each frame and draw objects using pretrained yolo_v2_tiny model.
	Finally, store drawed frames into 'out_video_path'
	"""
	reader, writer = open_video_with_opencv(in_video_path, out_video_path)
	yolo = yolov2tiny.YOLO_V2_TINY((416, 416, 3), "./y2t_weights.pickle", proc)

	acc_time, acc_cnt = 0, 0
	while reader.isOpened():
		okay, image = reader.read()
		if not okay:
			raise Exception(
			    "An error occured while reading from {}".format(in_video_path))

		image = resize_input(image)
		beg = datetime.datetime.now()
		batched_tensors = yolo.inference(image)
		end = datetime.datetime.now()
		inference_time = (end - beg).total_seconds()

		acc_time += inference_time
		acc_cnt += 1

		print(
		    "#{}: done in {} seconds\ttotal:{} seconds\tthroughput: {} frames per second"
		    .format(acc_cnt, inference_time, acc_time, acc_cnt / acc_time))

		for tensor in batched_tensors:
			proposals = yolov2tiny.postprocessing(tensor)
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
