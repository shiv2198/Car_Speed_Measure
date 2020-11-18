import cv2
import numpy
import pandas
import time
import tensorflow as tf
import caffe2 as cf

# def get_fps(start_time, frame_count):
# 	if frame_count >= 100:
# 		duration = float(time.time() - start_time)
# 		FPS = float(frame_count / duration)
#
# 		frame_count = 0
# 		start_time = time.time()
#
# 	else:
# 		frame_count += 1
#
# 	return (start_time,frame_count)



def draw_rect(image, box):
	height = image.shape[0]
	width = image.shape[1]
	y_min = int(max(1, (box[0] * height)))
	x_min = int(max(1, (box[1] * width)))
	y_max = int(min(height, (box[2] * height)))
	x_max = int(min(width, (box[3] * width)))

	# draw a rectangle on the image
	cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)


track_points = {"A":120,"B":160}


model_path = "/home/shivansh/Desktop/projects/Car_Speed_Measure/detect.tflite"


model = tf.lite.Interpreter(model_path=model_path)
input_details = model.get_input_details()
output_details = model.get_output_details()

print(input_details)
print(output_details)

model.allocate_tensors()

frame_num = 0
cap = cv2.VideoCapture('/home/shivansh/Desktop/projects/Car_Speed_Measure/car.mp4')
while True:
	ret, frame = cap.read()
	blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
								 ddepth=cv2.CV_8U)
	print("@@@@@@@@@@@@@@@@@@@@@@@@\n",blob,"\n@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
	image = cv2.resize(frame, (300, 300))
	frame_num += 1
	model.set_tensor(input_details[0]['index'], [image])
	model.invoke()
	print(output_details)
	rects = model.get_tensor(
		output_details[0]['index'])
	scores = model.get_tensor(
		output_details[2]['index'])
	classes = model.get_tensor(
		output_details[1]['index'])
	print(frame_num)
	print(rects, "\n##", scores)

	for index, score in enumerate(scores[0]):
		if score > 0.5 and classes[0][index] == 2:
			draw_rect(image, rects[0][index])

	cv2.imshow("image", image)
	# cv2.waitKey(0)


# image = cv2.imread("/home/shivansh/Desktop/Mask_RCNN_Matterport/images/car_in_city.jpg")
# image = cv2.resize(image, (300,300))

# model.set_tensor(input_details[0]['index'], [image])
# model.invoke()
# print(output_details)
# rects = model.get_tensor(
# 		output_details[0]['index'])
# scores = model.get_tensor(
# 		output_details[2]['index'])
# classes = model.get_tensor(
# 		output_details[1]['index'])
#
# print(rects,"\n##",scores)
#
# for index, score in enumerate(scores[0]):
# 		if score > 0.5 and classes[0][index]==2:
# 			draw_rect(image, rects[0][index])
#
# cv2.imshow("image", image)
# cv2.waitKey(0)

