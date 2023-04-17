import cv2


# Aceasta clasa a fost realizata cu scopul de a aplica modelul SSD (Single Shot-Detector) asupra stream-ului video,insa ulterior mi-am dat seama ca o solutie
# pentru a putea rezolva cerintele 2a,2b,2c ar fi sa introduc aceste linii de cod in zona in care s-ar aplica si modelul YoloV2,asa ca am pastrat clasa
# strict cu scop informativ,nicio metoda de aici nu este utilizata propriu-zis in program


class CarDetector:
    def __init__(self, model_path, config_path, labels_path, stream_url):
        self.model = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        with open(labels_path, 'rt') as f:
            self.labels = f.read().rstrip('\n').split('\n')
        self.stream_url = stream_url

    def run(self):
        cap = cv2.VideoCapture(self.stream_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)

            self.model.setInput(blob)
            detections = self.model.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.3:
                    # classId = int(detections[0, 0, i, 1])
                    x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                    y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                    x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                    y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('ssd', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


car_detector = CarDetector('frozen_inference_graph_V2.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
                           'coco.names.txt', 'http://87.57.111.162/mjpg/video.mjpg')
car_detector.detect_cars()
