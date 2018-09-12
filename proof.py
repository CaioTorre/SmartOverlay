import sys
import os

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from imageai.Detection import ObjectDetection
sys.stderr = stderr
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "media", "image.jpg"), output_image_path=os.path.join(execution_path, "media", "image_new.png"))

print("--------------------------------")
for eachObject in detections:
    print(eachObject["name"] + " : " + str(eachObject["percentage_probability"]))
    print("--------------------------------")

img = Image.open("image_new.png")
img.show()
