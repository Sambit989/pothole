import os
import argparse
import random

def gather(dataset_dir):
    imgs = []
    for f in os.listdir(dataset_dir):
        if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg'):
            imgs.append(os.path.join(dataset_dir,f))
    return sorted(imgs)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--out_dir', default='data')
    p.add_argument('--val_split', type=float, default=0.2)
    args = p.parse_args()

    imgs = gather(args.dataset_dir)
    random.shuffle(imgs)
    n_val = int(len(imgs)*args.val_split)
    val = imgs[:n_val]
    train = imgs[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,'train.txt'),'w') as f:
        for x in train: f.write(x+'\n')
    with open(os.path.join(args.out_dir,'valid.txt'),'w') as f:
        for x in val: f.write(x+'\n')
    print('Wrote', len(train), 'train and', len(val), 'valid images to', args.out_dir)
