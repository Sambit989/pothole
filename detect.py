import cv2
import numpy as np
import argparse
import time
from dimension_estimation import estimate_sizes_from_bboxes

def load_net(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect_image(net, image_path, conf_thresh=0.4, nms_thresh=0.4):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(out_names)
    boxes, confidences = [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > conf_thresh:
                center_x = int(detection[0]*W)
                center_y = int(detection[1]*H)
                w = int(detection[2]*W)
                h = int(detection[3]*H)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(conf))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    results = []
    if len(idxs)>0:
        for i in idxs.flatten():
            x,y,w,h = boxes[i]
            results.append({'bbox':[x,y,w,h],'conf':confidences[i]})
    return img, results

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--cfg', default='yolov4.cfg')
    p.add_argument('--weights', default='yolov4-pothole.weights')
    p.add_argument('--out', default='out.jpg')
    p.add_argument('--draw', action='store_true')
    args = p.parse_args()

    net = load_net(args.cfg, args.weights)
    img, results = detect_image(net, args.input)
    # draw
    for r in results:
        x,y,w,h = r['bbox']
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(img, f"pothole {r['conf']:.2f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
    cv2.imwrite(args.out, img)
    print('Detections:', results)
    # compute sizes if user provides calibration info interactively (not mandatory here)
    # Example usage: import dimension_estimation and call estimate_sizes_from_bboxes
