# yolo_lesson

## Быстрый старт

1. Установите [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Создайте и активируйте виртуальное окружение (по умолчанию появится `.venv`):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. Установите проектные зависимости (используются `pyproject.toml` и `uv.lock`):
   ```bash
   uv sync
   ```

После синхронизации зависимости доступны через `uv run`. `runs/` с результатами обучения храните в репозитории.

## Обучение модели

- Откройте ноутбук `yolov8_persons.ipynb` и запустите его шаги, чтобы получить веса `best.pt`.
- При необходимости запустите Jupyter напрямую:
  ```bash
  uv run jupyter notebook
  ```

## Запуск детекции с веб-камеры

```bash
uv run python webcam_yolov8_ultra.py \
  --model runs/detect/train/weights/best.pt \
  --camera 0 \
  --device mps
```

- `--model` — путь к весам модели в формате `.pt`;
- `--camera` — индекс локальной камеры (`0` по умолчанию);
- `--rtsp` — адрес RTSP-потока (если используете IP-камеру);
- `--device` — устройство расчёта: `cuda` (GPU), `cpu`, либо `mps` (Apple Silicon).

## Jetson

Экспортируйте модель в TensorRT engine:

```bash
uv run yolo export model=runs/detect/train/weights/best.pt format=engine half=True device=0
```

Запустите inference:

```bash
uv run python webcam_yolov8_ultra.py \
  --model runs/detect/train/weights/best.engine \
  --rtsp "rtsp://admin:PSWD@192.168.31.8:554/Streaming/Channels/101" \
  --device cuda
```
