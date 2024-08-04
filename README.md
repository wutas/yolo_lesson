# yolo_lesson

Создайте выделенное оркужения. 
Установите зависимости:

```
pip install -r requirements.txt
```

Используйте терминальный запуск скприта:

```
python webcam_yolov8_ultra.py --model runs/detect/train/weights/best.pt --device mps
```

model - путь до весов модели в формте .pt
device - на чем будет идати расчет: cuda (GPU), cpu (процессор) или mps (M... процессор от Apple)
