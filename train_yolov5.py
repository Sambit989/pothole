# Train using Ultralytics YOLOv5 on the JayGala dataset (one-class)
# This script expects you have cloned yolov5 repo or will clone it.
# It prepares a YAML dataset descriptor and launches training via python train.py
import os, yaml, argparse, shutil

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='dataset/jaygala', help='Path to JayGala dataset (images & labels)')
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=16)
args = parser.parse_args()

# Create dataset yaml for yolov5
data_yaml = {
    'train': os.path.join(args.dataset_dir, 'images', 'train'),
    'val': os.path.join(args.dataset_dir, 'images', 'val'),
    'nc': 1,
    'names': ['pothole']
}
os.makedirs('yolov5_dataset', exist_ok=True)
with open('yolov5_dataset/data.yaml','w') as f:
    yaml.dump(data_yaml, f)
print('Wrote yolov5 data descriptor to yolov5_dataset/data.yaml')
print('To train: clone https://github.com/ultralytics/yolov5 and run:')
print('python train.py --img', args.imgsz, '--batch', args.batch, '--epochs', args.epochs, '--data yolov5_dataset/data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --name pothole_yolov5')
