"""Code reference
https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
"""

import torchvision
import numpy as np

class CIFAR10LT(torchvision.datasets.CIFAR10):
    def __init__(
        self, root: str, train: bool = True, download: bool = True, transform = None, 
        imbalance_type: str = None, imbalance_factor: int = 1):
        
        super(CIFAR10LT, self).__init__(root=root, train=train, transform=transform, download=download)
        self.num_classes = len(np.unique(self.targets))
        self.imbalance_type = imbalance_type
        self.imbalance_factor = imbalance_factor

    def _get_img_num_per_cls(self, imb_type: str, imb_factor: float):
        gamma = 1. / imb_factor
        img_max = len(self.data) / self.num_classes
        
        img_num_per_cls = []
        
        # imbalance type
        if imb_type == 'exp':
            for cls_idx in range(self.num_classes):
                num = img_max * (gamma ** (cls_idx / (self.num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        
        elif imb_type == 'step':
            for cls_idx in range(self.num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.num_classes // 2):
                img_num_per_cls.append(int(img_max * gamma))
        
        else:
            img_num_per_cls.extend([int(img_max)] * self.num_classes)
        
        # make dictionary about the number of images per class
        num_per_cls = dict([(c_i, img_num_c_i) for c_i, img_num_c_i in zip(np.unique(self.targets), img_num_per_cls)])
        setattr(self, 'num_per_cls', num_per_cls)

    def gen_imbalanced_data(self):
        # get the number of images per class
        self._get_img_num_per_cls(imb_type=self.imbalance_type, imb_factor=self.imbalance_factor)
        
        imb_data = []
        imb_targets = []
        
        for c_i, img_num_c_i in self.num_per_cls.items():
            # find sample indexes for class i
            c_idx = np.where(np.array(self.targets, dtype=np.int64) == c_i)[0]
            
            # shuffle
            np.random.shuffle(c_idx)
            
            # select `img_num_c_i` samples for class i 
            selected_c_idx = c_idx[:img_num_c_i]
            
            # select 
            imb_data.append(self.data[selected_c_idx, ...])
            imb_targets.extend([c_i, ] * img_num_c_i)
            
        imb_data = np.vstack(imb_data)
        setattr(self, 'data', imb_data)
        setattr(self, 'targets', imb_targets)


class CIFAR100LT(CIFAR10LT):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }