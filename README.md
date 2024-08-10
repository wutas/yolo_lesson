# yolo_lesson
Обучите модель из ipynb файла и сохраните папку runs.

Создайте выделенное оркужения. 
Установите зависимости:

```
pip install -r requirements.txt
```

Обучлите модель в ноутбуке: yolov8_persons

Используйте терминальный запуск скприта:

```
python webcam_yolov8_ultra.py --model runs/detect/train/weights/best.pt --device mps
```

model - путь до весов модели в формте .pt

device - на чем будет идати расчет: cuda (GPU), cpu (процессор) или mps (M... процессор от Apple)
