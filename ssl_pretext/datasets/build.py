from torch.utils.data import Dataset
import torch

import random


class RotationDataset(Dataset):
    def __init__(self, dataset: Dataset, is_train: bool = True):
        super().__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.rotations = [0,1,2,3]
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx] 
        
        if self.is_train:
            target = random.choice(self.rotations)
            input = torch.rot90(input=img, k=target, dims=[1,2])

        else:        
            target = torch.LongTensor(self.rotations)
            input = [img]
            for t in target[1:]:
                input.append(torch.rot90(input=img, k=t, dims=[1,2]))
            input = torch.stack(input)

        return input, target