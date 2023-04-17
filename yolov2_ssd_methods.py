import cv2


class VideoCapture:
    def __init__(self, detector, url, ssd_model_path, ssd_config_path, labels_path):
        self.yolo_detector = detector
        self.cap1 = cv2.VideoCapture(url)

        self.cap2 = cv2.VideoCapture(url)
        self.ssd_model = cv2.dnn.readNetFromTensorflow(ssd_model_path, ssd_config_path)
        with open(labels_path, 'rt') as f:
            self.labels = f.read().rstrip('\n').split('\n')

    def ssd_method(self, ssd_frame):
        blob = cv2.dnn.blobFromImage(ssd_frame, size=(300, 300), swapRB=True, crop=False)

        self.ssd_model.setInput(blob)
        detections = self.ssd_model.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                x1 = int(detections[0, 0, i, 3] * ssd_frame.shape[1])
                y1 = int(detections[0, 0, i, 4] * ssd_frame.shape[0])
                x2 = int(detections[0, 0, i, 5] * ssd_frame.shape[1])
                y2 = int(detections[0, 0, i, 6] * ssd_frame.shape[0])
                cv2.rectangle(ssd_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self):
        while True:
            yolo_ret, yolo_frame = self.cap1.read()
            ssd_ret, ssd_frame = self.cap1.read()
            if not yolo_ret or not ssd_ret:
                print("Could not read frame")
                break

            yolo_frame = cv2.resize(yolo_frame, (640, 360))
            boxes = self.yolo_detector.detection_algorithm(yolo_frame)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(yolo_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ssd_frame = cv2.resize(ssd_frame, (640, 360))
            self.ssd_method(ssd_frame)

            side_by_side = cv2.hconcat([yolo_frame, ssd_frame])
            cv2.imshow('Two videos', side_by_side)
            if cv2.waitKey(1) == ord('q'):
                return
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
