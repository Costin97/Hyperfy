import cv2
import numpy as np
from yolov2_ssd_methods import VideoCapture


class YoloV2Model:
    def __init__(self, model_path, weights_path):
        self.model = cv2.dnn.readNetFromDarknet(model_path, weights_path)
        self.output_layers = self.get_output_layers()

    def get_output_layers(self):
        # layer_names = self.model.getLayerNames()
        output_layers = []
        leng = self.model.getUnconnectedOutLayers()
        for i in range(int(leng)):
            output_layers.append(i)
        return output_layers

    def detection_algorithm(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward(self.output_layers)

        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.3:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append((x, y, width, height))
        return boxes


camera_urls = ["http://195.196.36.242/mjpg/video.mjpg", "http://23.25.62.174/mjpg/video.mjpg",
               "http://87.139.9.247/mjpg/video.mjpg",
               "http://166.247.77.253:82/mjpg/video.mjpg", "http://173.219.84.12:8000/mjpg/video.mjpg",
               "http://166.251.105.190/mjpg/video.mjpg", "http://185.137.146.14/mjpg/video.mjpg",
               "http://212.112.136.4:83/mjpg/video.mjpg", "http://166.154.166.8/mjpg/video.mjpg",
               "http://87.57.111.162/mjpg/video.mjpg"]

yolo_detector = YoloV2Model("Models/yolov2-voc.cfg.txt", "Models/yolov2-voc.weights")
video_detector = VideoCapture(yolo_detector, camera_urls[6], 'Models/frozen_inference_graph_V2.pb',
                              'Models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
                              'Models/coco.names.txt')
video_detector.run()
