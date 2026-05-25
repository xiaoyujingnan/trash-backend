web: YOLO_CONFIG_DIR=/tmp/Ultralytics OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NUM_THREADS=2 gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
