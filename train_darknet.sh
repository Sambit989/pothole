#!/bin/bash
# Instructions to train using AlexeyAB Darknet (Linux)
# 1. Clone darknet:
#    git clone https://github.com/AlexeyAB/darknet.git
# 2. Edit Makefile to enable GPU=1, CUDNN=1 if you have CUDA. Then:
#    make
# 3. Place this repo files in darknet root or update paths below.
# 4. Create 'data/obj' directory and copy images and labels (YOLO .txt) from JayGala dataset.
# 5. Create 'obj.data' and 'obj.names' (provided).
# 6. Start training:
#    ./darknet detector train obj.data yolov4.cfg yolov4.conv.137 -dont_show -map
#
# To resume from backup:
#    ./darknet detector train obj.data yolov4.cfg backup/yolov4-pothole_last.weights -map
#
echo 'See README.md for full instructions.'
