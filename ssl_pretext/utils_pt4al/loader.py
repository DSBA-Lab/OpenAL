from torch.utils.data import Dataset
import torch

import random


class RotationDataset(Dataset):
    def __init__(self, dataset, is_train=True, transform=None):
        self.dataset = dataset
        self.is_train = is_train
        self.transform = transform
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx] 
        
        img1, img2, img3 = torch.rot90(img, 1, [1,2]), torch.rot90(img, 2, [1,2]), torch.rot90(img, 3, [1,2])
        imgs = [img, img1, img2, img3]
        rotations = [0,1,2,3]
        random.shuffle(rotations)

        if self.is_train:
            return [imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]]], [rotations[0], rotations[1], rotations[2], rotations[3]]
        else:
            return [imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]]], [rotations[0], rotations[1], rotations[2], rotations[3]], idx