import os
from PIL import Image
from torch.utils.data import Dataset 


class WaterBird(Dataset):
    def __init__(self, datadir: str, meta_info: str, transform = None):
        
        # class
        self.classes = ["land-bird", "water-bird"]
        self.class_to_idx = dict([(c, i) for i, c in enumerate(self.classes)])

        # place
        self.place_to_idx = {"land": 0, "water": 1}

        self.datadir = datadir
        self.meta_info = meta_info
        self.transform = transform
        
    def __getitem__(self, i):
        info_i = self.meta_info.iloc[i]
        img = Image.open(os.path.join(self.datadir, info_i['img_filename'])).convert('RGB')
        
        if self.transform != None:
            img = self.transform(img)
            
        target = info_i['y']
        
        return img, target