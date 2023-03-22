import pandas as pd
import os
import cv2
from torch.utils.data import Dataset


class ALDataset(Dataset):
    def __init__(self, datadir: str, name: str, transform: list):
        
        self.datadir = datadir
        self.label_info = pd.read_csv(os.path.join(datadir, name))
        self.data_info = self.label_info
        
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx]['img_path']
        target = self.data_info.iloc[idx]['label']
        
        img = cv2.imread(os.path.join(self.datadir, img_path))
        img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.data_info)