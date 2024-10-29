import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image
from sklearn.model_selection import train_test_split

# class DomainNet(Dataset):
#     def __init__(self, root: str, train: bool = True, in_domain: str = 'real', transform = None, cls_topk = 50):
#         super(DomainNet, self).__init__()
#         self.domains = [
#             'clipart', 
#             'infograph', 
#             'painting', 
#             'quickdraw', 
#             'real', 
#             'sketch'
#         ]
#         self.train = train
#         self.root = root
#         self.in_domain = in_domain
#         self.cls_topk = cls_topk
#         self.get_file_info(cls_topk=self.cls_topk)
        
#         self.transform = transform
        
#     def get_file_info(self, cls_topk = 50):
#         if self.train:
#             file_info = []
#             for d in self.domains:
#                 info_i = pd.read_csv(
#                     os.path.join(self.root, f'{d}_train.txt'), sep=' ', header=None
#                 )
#                 info_i['domain'] = d
#                 file_info.append(info_i)

#             file_info = pd.concat(file_info, axis=0)
#         else:
#             file_info = pd.read_csv(f'/datasets/DomainNet/{self.in_domain}_test.txt', sep=' ', header=None)
#             file_info['domain'] = self.in_domain
            
#         file_info.columns = ['filepath', 'class_idx', 'domain']
#         file_info = file_info.reset_index(drop=True)
#         # select topk class index
#         file_info = self.select_cls_topk(file_info=file_info, cls_topk=cls_topk)
        
#         # set classes
#         file_info['classes'] = file_info['filepath'].apply(lambda x: x.split('/')[1])
#         classes = [f'{self.in_domain} {c}' for c in file_info['classes'].unique()]
        
#         # set file path
#         file_info['filepath'] = file_info['filepath'].apply(lambda x: os.path.join(self.root, x))
        
#         # set OOD class
#         nb_classes = file_info['class_idx'].unique()
#         file_info.loc[file_info['domain'] != self.in_domain, 'class_idx'] = len(nb_classes)
#         targets = file_info['class_idx'].values
        
#         # set attr
#         setattr(self, 'classes', classes)
#         setattr(self, 'targets', targets)
#         setattr(self, 'file_info', file_info)
        
#     def select_cls_topk(self, file_info, cls_topk):
#         if cls_topk == -1:
#             return file_info
#         else:
#             cls_topk_idx = file_info[file_info['domain']==self.in_domain]['class_idx'].value_counts().index[:cls_topk]
#             file_info = file_info[file_info['class_idx'].isin(cls_topk_idx)]
#             file_info = file_info.reset_index(drop=True)
            
#             target_map = dict(zip(cls_topk_idx, np.arange(cls_topk)))
    
#             new_targets = []
#             targets = file_info['class_idx'].tolist()
#             for t in targets:
#                 new_targets.append(target_map[t])
                
#             file_info.loc[:,'class_idx'] = np.array(new_targets)
#         return file_info
        
#     def __getitem__(self, i):
#         info_i = self.file_info.iloc[i]
        
#         img = Image.open(info_i['filepath']).convert('RGB')
#         if self.transform != None:
#             img = self.transform(img)
            
#         target = info_i['class_idx']
        
#         return img, target
        
        
#     def __len__(self):
#         return len(self.file_info)
    
class DomainNet(Dataset):
    def __init__(self, root: str, train: bool = True, in_domain: str = 'real', transform = None, use_domain: bool = False, return_features: bool = False):
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
        
        self.transform = transform
        self.use_domain = use_domain
        self.return_features = return_features
        
        self.get_file_info()
        
    def get_file_info(self):
        if self.train:
            # file_info = pd.read_csv(os.path.join(self.root, f'{self.in_domain}_train.txt'), sep=' ', header=None)
            # file_info.columns = ['filepath', 'class_idx']
            
            # # set classes
            # file_info['classes'] = file_info['filepath'].apply(lambda x: x.split('/')[1])
            
            # if self.in_domain == 'painting':
            #     file_info = file_info[~file_info['classes'].isin(['t-shirt', 'syringe'])]
                
            #     old_class_idx = file_info['class_idx'].unique()
            #     class_idx_map = {}
            #     for i, oc in enumerate(old_class_idx):
            #         class_idx_map[oc] = i
                
            #     file_info['class_idx'] = file_info['class_idx'].map(class_idx_map)
            #     file_info = file_info.reset_index(drop=True)
            
            file_info = []
            for d in self.domains:
                info_i = pd.read_csv(
                    os.path.join(self.root, f'{d}_train.txt'), sep=' ', header=None
                )
                info_i['domain'] = d            
                file_info.append(info_i)

            file_info = pd.concat(file_info, axis=0)
            file_info.columns = ['filepath', 'class_idx', 'domain']
            file_info.loc[file_info['domain']!=self.in_domain, 'class_idx'] = file_info['class_idx'].nunique()
            
            # set classes
            file_info['classes'] = file_info['filepath'].apply(lambda x: x.split('/')[1])
            
            # for painting
            if self.in_domain == 'painting':
                file_info = file_info[~file_info['classes'].isin(['t-shirt', 'syringe'])]
                
                old_class_idx = file_info.loc[file_info['domain']==self.in_domain, 'class_idx'].unique()
                class_idx_map = {}
                for i, oc in enumerate(old_class_idx):
                    class_idx_map[oc] = i
                
                file_info.loc[file_info['domain']==self.in_domain, 'class_idx'] = file_info.loc[file_info['domain']==self.in_domain, 'class_idx'].map(class_idx_map)
                file_info.loc[file_info['domain']!=self.in_domain, 'class_idx'] = len(old_class_idx)
            
            file_info = file_info.reset_index(drop=True)
            
        else:
            file_info = pd.read_csv(os.path.join(self.root, f'{self.in_domain}_test.txt'), sep=' ', header=None)
            file_info.columns = ['filepath', 'class_idx']
        
            # set classes
            file_info['classes'] = file_info['filepath'].apply(lambda x: x.split('/')[1])
        
            # for painting
            if self.in_domain == 'painting':
                file_info = file_info[~file_info['classes'].isin(['t-shirt', 'syringe'])]
                
                old_class_idx = file_info['class_idx'].unique()
                class_idx_map = {}
                for i, oc in enumerate(old_class_idx):
                    class_idx_map[oc] = i
                    
                file_info['class_idx'] = file_info['class_idx'].map(class_idx_map)
                file_info = file_info.reset_index(drop=True)
        
        # set classes
        if self.use_domain:
            classes = [f'{self.in_domain} {c}' for c in file_info['classes'].unique()]
        else:
            classes = file_info['classes'].unique()
        
        
        # set file path
        file_info['filepath'] = file_info['filepath'].apply(lambda x: os.path.join(self.root, x))
        targets = file_info['class_idx'].values
        
        # set attr
        setattr(self, 'classes', classes)
        setattr(self, 'targets', targets)
        setattr(self, 'file_info', file_info)
    
        
    def get_features(self, savedir, modelname, is_train):
        features = torch.load(
            os.path.join(
                savedir, f'DomainNet-{self.in_domain}', modelname,
                f"{'train' if is_train else 'test'}_features.pt"
            )
        )
        
        setattr(self, 'features', features)
        
    def __getitem__(self, i):
        info_i = self.file_info.iloc[i]
        
        if self.return_features:
            img = self.features[i]
        else:
            img = Image.open(info_i['filepath']).convert('RGB')
            if self.transform != None:
                img = self.transform(img)
            
        target = self.targets[i]
        
        return img, target
        
        
    def __len__(self):
        return len(self.file_info)