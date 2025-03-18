import os
import shutil
import subprocess

subprocess.run(['curl', '-L', '-o', 'tiny-imagenet-200.zip', 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'])
subprocess.run(['unzip', 'tiny-imagenet-200.zip', '-d', './dataset/tiny-imagenet'])
subprocess.run(['rm', 'tiny-imagenet-200.zip'])

with open('./dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'./dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'./dataset/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'./dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('./dataset/tiny-imagenet/tiny-imagenet-200/val/images')