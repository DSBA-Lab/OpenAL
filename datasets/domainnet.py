import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class DomainNet(Dataset):
    def __init__(self, root: str, train: bool = True, in_domain: str = 'real', transform = None):
        super(DomainNet, self).__init__()
        self.domains = [
            'clipart', 
            'infograph', 
            'painting', 
            'quickdraw', 
            'real', 
            'sketch'
        ]
        self.train = train
        self.root = root
        self.in_domain = in_domain
        self.get_file_info()
        
        self.transform = transform
        
    def get_file_info(self):
        if self.train:
            file_info = []
            for d in self.domains:
                info_i = pd.read_csv(
                    os.path.join(self.root, f'{d}_train.txt'), sep=' ', header=None
                )
                info_i['domain'] = d
                file_info.append(info_i)

            file_info = pd.concat(file_info, axis=0)
        else:
            file_info = pd.read_csv(f'/datasets/DomainNet/{self.in_domain}_test.txt', sep=' ', header=None)
            
        file_info.columns = ['filepath', 'class_idx', 'domain']
        file_info = file_info.reset_index(drop=True)
        
        # set classes
        file_info['classes'] = file_info['filepath'].apply(lambda x: x.split('/')[1])
        classes = [f'{self.in_domain} {c}' for c in file_info['classes'].unique()]
        
        # set file path
        file_info['filepath'] = file_info['filepath'].apply(lambda x: os.path.join(self.root, x))
        
        # set OOD class
        nb_classes = file_info['class_idx'].unique()
        file_info.loc[file_info['domain'] != self.in_domain, 'class_idx'] = len(nb_classes)
        targets = file_info['class_idx'].values
        
        # set attr
        setattr(self, 'classes', classes)
        setattr(self, 'targets', targets)
        setattr(self, 'file_info', file_info)
        
    def __getitem__(self, i):
        info_i = self.file_info.iloc[i]
        
        img = Image.open(info_i['filepath']).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
            
        target = info_i['class_idx']
        
        return img, target
        
        
    def __len__(self):
        return len(self.file_info)