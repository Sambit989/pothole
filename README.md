# Pothole Detection & Dimension Estimation (Full Functional Project)

## Summary
This project implements the system from the paper *Pothole Detection and Dimension Estimation System using Deep Learning (YOLO) and Image Processing*.
The project is configured to train a YOLOv4 detector (one-class 'pothole') and estimate pothole dimensions using triangular similarity.

## What I prepared for you
- Darknet/YOLOv4-ready config (`yolov4.cfg`) and helper scripts for Darknet training.
- Colab notebook (`train_darknet_colab.ipynb`) that compiles AlexeyAB darknet in Colab and launches training.
- `train_yolov5.py` helper to train using Ultralytics YOLOv5 if you prefer PyTorch.
- `detect.py` using OpenCV DNN for inference.
- `dimension_estimation.py` for focal length and size estimation.
- `app.py` Flask web app for image upload, detection, and size estimation. Templates included.
- `download_and_prepare.sh` helper to clone JayGala dataset and fetch pretrained conv weights.
- `prepare_dataset.py` to make train/valid lists.
- `calibrate/` tools to compute perceived focal length.

## How to run (minimal testing / inference)
1. Place your trained YOLOv4 weights at the project root as `yolov4-pothole.weights`. If you don't have trained weights yet, train using the Colab notebook.
2. Install dependencies:
   ```
   pip install -r requirements.txt flask opencv-python-headless numpy torch torchvision
   ```
   (For full training you will need CUDA-enabled PyTorch or run in Colab.)
3. Start the Flask app:
   ```
   python app.py
   ```
   Visit `http://localhost:5000` to upload images and run detection. Provide the perceived focal length (F) in pixels for size estimation; if unknown, follow the `calibrate/README.md` steps using a ruler at known distances.

## Training
- For best fidelity with the paper, use AlexeyAB Darknet (YOLOv4). Use the provided Colab notebook `train_darknet_colab.ipynb`.
- Alternatively, use Ultralytics YOLOv5 via `train_yolov5.py` helper to create dataset YAML and then train using the `ultralytics/yolov5` repo.

## Limitations & notes
- I could not train the model here (no long GPU runs). This repo contains everything to train on your machine or Colab.
- After training, place the resulting `.weights` (Darknet) or `.pt` (PyTorch) in the repo and use `detect.py` or the Flask app.

## Files
- `yolov4.cfg`, `obj.names`, `obj.data` — Darknet config and metadata
- `prepare_dataset.py` — prepare train/valid lists
- `train_darknet_colab.ipynb` — ready-to-run Colab notebook for Darknet training
- `train_yolov5.py` — helper for YOLOv5 training
- `detect.py` — inference using OpenCV DNN
- `dimension_estimation.py` — size estimation
- `app.py`, `templates/index.html` — Flask demo
- `calibrate/` — calibration helpers

## If you want, I can:
- Populate the Colab notebook with checkpoint upload/download steps and automatic evaluation plotting.
- Convert the detect pipeline to accept video streams (webcam or RTSP).
- Add Dockerfile to run the Flask app inside a container.

