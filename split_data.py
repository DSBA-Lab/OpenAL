import os
import pandas as pd
import numpy as np
import cv2
import math

from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def split_data(datadir: str, seed: int = 42, ratio: list = [0.7, 0.2, 0.1]):
    assert len(ratio) == 3, "ratio must be length 3 (train, validation, test)"
    assert math.fsum(ratio) == 1.0, "sum of ratio must be 1"
    
    # ratio
    dataset_ratio = dict(zip(['train','valid','test'], ratio))
    
    # read file path
    file_path = glob(os.path.join(datadir, '**/*.png'), recursive=True)
    
    # total file length
    total_size = len(file_path)
    
    # split size
    dataset_size = {}
    dataset_size['train'] = int(total_size * dataset_ratio['train'])
    dataset_size['valid'] = int(total_size * dataset_ratio['valid'])
    dataset_size['test'] = int(total_size - dataset_size['train'] - dataset_size['valid'])

    print('train size: ',dataset_size['train'])
    print('valid size: ',dataset_size['valid'])
    print('test size: ',dataset_size['test'])
    print()

    # split datasets
    df = pd.DataFrame({
        'img_path': list(map(lambda x: '/'.join(x.split('/')[3:]), file_path)),
        'label': list(map(lambda x: int(x.split('/')[3]), file_path))
    })

    train, test = train_test_split(df, train_size=dataset_size['train'], stratify=df['label'], random_state=seed)
    valid, test = train_test_split(test, train_size=dataset_size['valid'], stratify=test['label'], random_state=seed)

    # save datasets
    train.to_csv(os.path.join(datadir, f'train_seed{seed}.csv'), index=False)
    valid.to_csv(os.path.join(datadir, f'validation_seed{seed}.csv'), index=False)
    test.to_csv(os.path.join(datadir, f'test_seed{seed}.csv'), index=False)

def load_images(datadir: str, file_path: list, img_size: list):
    imgs = []
    for p in file_path:
        img = cv2.resize(cv2.imread(os.path.join(datadir,p)), dsize=(img_size))
        imgs.append(img)

    return np.array(imgs)

def check_split_info(datadir: str, seed: list, img_size: list = [224,224]):
    for s in seed:
        # read datasets
        train = pd.read_csv(os.path.join(datadir, f'train_seed{s}.csv'))
        valid = pd.read_csv(os.path.join(datadir, f'validation_seed{s}.csv'))
        test = pd.read_csv(os.path.join(datadir, f'test_seed{s}.csv'))

        print('='*20)
        print('train.shape: ',train.shape)
        print('valid.shape: ',valid.shape)
        print('test.shape: ',test.shape)
        print()
        
        # statistics
        train_imgs = load_images(datadir=datadir, file_path=train.img_path, img_size=img_size)

        print('mean: ',(train_imgs / 255.).mean(axis=(0,1,2)))
        print('std: ',(train_imgs / 255.).std(axis=(0,1,2)))
        
        # figure
        train_value = train.label.value_counts().reset_index()
        valid_value = valid.label.value_counts().reset_index()
        test_value = test.label.value_counts().reset_index()

        fig, ax = plt.subplots(1, 3, figsize=(10,3))

        title = ['Train set','Validation set', 'Test set']
        for i, d in enumerate([train_value, valid_value, test_value]):
            sns.barplot(
                x = 'index',
                y = 'label',
                data = d,
                ax = ax[i]
            )
            ax[i].set_ylim([0, d['label'].max()+2000])
            ax[i].set_ylabel('Frequency')
            ax[i].set_xlabel('Clsss')
            ax[i].set_title(title[i] + f'seed {s}')

            for container in ax[i].containers:
                ax[i].bar_label(container, fmt='%d', size=13)

        plt.tight_layout()
        plt.show()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='data directory')
    parser.add_argument('--seed', type=int, nargs='+', help='seed numbers')
    parser.add_argument('--ratio', type=float, nargs='+', help='split ratio (train, validation, test)')
    args = parser.parse_args()
    
    for s in args.seed:
        split_data(datadir=args.datadir, seed=s, ratio=args.ratio)