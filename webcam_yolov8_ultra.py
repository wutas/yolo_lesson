import cv2
import argparse
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path, device='cpu', imgsz=640, conf_threshold=0.25, overlap_threshold=0.45):
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.overlap_threshold = overlap_threshold

    def detect(self, frame):
        self.original_shape = frame.shape
        resized_frame = cv2.resize(frame, (self.imgsz, self.imgsz))
        results = self.model(resized_frame, conf=self.conf_threshold, iou=self.overlap_threshold, imgsz=self.imgsz, device=self.device)
        return results

    def draw_detections(self, frame, results):
        height_ratio = self.original_shape[0] / self.imgsz
        width_ratio = self.original_shape[1] / self.imgsz

        for detection in results:
            for bbox in detection.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = int(x1 * width_ratio)
                y1 = int(y1 * height_ratio)
                x2 = int(x2 * width_ratio)
                y2 = int(y2 * height_ratio)
                conf = detection.boxes.conf[0]
                cls = detection.boxes.cls[0]
                label = f'{self.model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


class WebcamYOLOv8:
    def __init__(self, model_path, device='cpu'):
        self.detector = YOLOv8Detector(model_path, device)
        self.cap = cv2.VideoCapture(0)

    def start(self):
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.detector.detect(frame)
            frame = self.detector.draw_detections(frame, results)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Webcam Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLOv8 model file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to run the model on (cpu, cuda or Apple Sil.)')
    
    args = parser.parse_args()
    
    model_path = args.model
    device = args.device
    
    webcam_yolov8 = WebcamYOLOv8(model_path, device)
    webcam_yolov8.start()

