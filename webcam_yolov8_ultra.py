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
        if not results:
            return frame

        height_ratio = self.original_shape[0] / self.imgsz
        width_ratio = self.original_shape[1] / self.imgsz

        for detection in results:
            if detection.boxes is None:
                continue

            for box in detection.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 * width_ratio)
                y1 = int(y1 * height_ratio)
                x2 = int(x2 * width_ratio)
                y2 = int(y2 * height_ratio)

                conf = float(box.conf[0]) if box.conf is not None else 0.0
                cls_idx = int(box.cls[0]) if box.cls is not None else -1

                label_name = str(cls_idx)
                if isinstance(self.model.names, dict):
                    label_name = self.model.names.get(cls_idx, label_name)
                elif isinstance(self.model.names, list) and 0 <= cls_idx < len(self.model.names):
                    label_name = self.model.names[cls_idx]

                label = f'{label_name} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


class VideoYOLOv8:
    def __init__(self, model_path, source, device='cpu', is_rtsp=False):
        self.detector = YOLOv8Detector(model_path, device)
        self.source = source
        self.is_rtsp = is_rtsp

        self.cap = cv2.VideoCapture(self.source)

        if self.is_rtsp:
            # Настраиваем параметры для более стабильного RTSP потока
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)

    def start(self):
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return

        source_type = "RTSP stream" if self.is_rtsp else "camera"
        print(f"Successfully connected to {source_type}: {self.source}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame. Reconnecting...")
                if self.is_rtsp:
                    # Попытка переподключения к RTSP
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    if not self.cap.isOpened():
                        print("Reconnection failed. Exiting.")
                        break
                    continue
                else:
                    print("Unable to read from camera. Exiting.")
                    break

            # Обработка кадра
            results = self.detector.detect(frame)
            frame = self.detector.draw_detections(frame, results)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 video detection')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLOv8 model file')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    parser.add_argument('--camera', type=int, default=0, help='Index of the local camera to use when RTSP is not set')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to run the model on')
    
    args = parser.parse_args()

    source = args.rtsp if args.rtsp else args.camera
    is_rtsp = args.rtsp is not None

    video_detector = VideoYOLOv8(
        model_path=args.model,
        source=source,
        device=args.device,
        is_rtsp=is_rtsp
    )
    video_detector.start()
