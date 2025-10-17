import os
import shutil

# Paths
src_dir = 'dataset/Pothole Dataset'
train_txt = 'data/train.txt'
val_txt = 'data/valid.txt'
img_train_dir = os.path.join(src_dir, 'images/train')
img_val_dir = os.path.join(src_dir, 'images/val')
label_train_dir = os.path.join(src_dir, 'labels/train')
label_val_dir = os.path.join(src_dir, 'labels/val')

os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

def move_files(txt_file, img_dest, label_dest):
    with open(txt_file, 'r') as f:
        for line in f:
            img_name = os.path.basename(line.strip())
            img_src = os.path.join(src_dir, img_name)
            label_src = img_src.replace('.jpg', '.txt')
            # Move image
            if os.path.exists(img_src):
                shutil.copy(img_src, os.path.join(img_dest, img_name))
            # Move label
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(label_dest, os.path.basename(label_src)))

move_files(train_txt, img_train_dir, label_train_dir)
move_files(val_txt, img_val_dir, label_val_dir)

print('Dataset reorganized for YOLOv5 training.')
