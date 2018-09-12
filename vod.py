import sys
import os

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from imageai.Detection import ObjectDetection
sys.stderr = stderr
from PIL import Image
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
vidcap = cv2.VideoCapture(os.path.join(execution_path, "media", "src.mp4"))
success, image = vidcap.read()
count = 0
while success:
    thisFrame = np.asarray(image)
    detections = detector.detectObjectsFromImage(input_image=thisFrame, input_type="array", output_image_path=os.path.join(execution_path, "out_vod", "frame_out" + str(count) + ".jpg"))

    for eachObject in detections:
        print("F" + str(count) + " - " + eachObject["name"] + " : " + str(eachObject["percentage_probability"]))
    print("--------------------------------------------")

    success,image = vidcap.read()
    count += 1
