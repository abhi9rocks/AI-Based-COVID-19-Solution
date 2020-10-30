# Importing the image AI library
# this library is an API used object detection system 
from imageai.Detection.Custom import CustomVideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
# using openCV to start the camera for taking the video feed
camera = cv2.VideoCapture(0)

detector = CustomVideoObjectDetection()
# our model is trained using transfer learning on YOLOV3.
detector.setModelTypeAsYOLOv3()
# put your trained keras model path in the space provided.
detector.setModelPath("Path to trained keras model")
# put the configuration file's path over here
detector.setJsonPath("path to json configuration file")
detector.loadModel()
#here the video is taken from the camera and in the outpat path, put your path where you can access the output video feed.
#minimum_percentage_probability : you can increase or decrease to fine tune your output.
detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "name of output video"),
                                          frames_per_second=16,
                                          minimum_percentage_probability=40,
                                          log_progress=True)