"""
    加载数据集
"""

import torch
import torchvision

class Dataset(torch.utils.data.dataset.Dataset):
    """
        自定义数据
        ---
    """
    def __init__(self):
        pass


    def __len__(self):
        """
            返回整个数据集的大小
        """
        return self.len
    def __getitem__(self, index):
        """
            获取一些索引的数据，使dataset[i]返回数据集中第i个样本。
        """
        assert index <= len(self), 'index range error'