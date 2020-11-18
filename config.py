import tensorflow as tf
import cv2

# Camera caliberations
focal_length = 0
camera_width = 0
image_heigth = 0
image_width  = 0


# Image Capture Configurations
track_length = 200

tracking_frames = 5

track_points = {"A":120,"B":160}

real_distance  = 0


model = ""
confidence = 50


