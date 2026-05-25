web: YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180
