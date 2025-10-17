#!/bin/bash
# Script to download JayGala dataset (GitHub) and prepare for darknet training
set -e
echo "Cloning JayGala dataset..."
if [ ! -d dataset/jaygala ]; then
  git clone https://github.com/jaygala24/pothole-detection.git dataset/jaygala
else
  echo "Dataset already cloned."
fi
echo "Copying images & labels to darknet/data/obj..."
mkdir -p darknet/data/obj
# User should run this script from project root where darknet is present or adjust paths
# This is a helper; manual verification recommended.
cp -v dataset/jaygala/images/* darknet/data/obj/ || true
cp -v dataset/jaygala/labels/* darknet/data/obj/ || true
echo "Download pretrained conv weights (yolov4.conv.137)..."
wget -O darknet/yolov4.conv.137 https://pjreddie.com/media/files/yolov4.conv.137 || true
echo "Done. Verify files and run training as instructed in README."
