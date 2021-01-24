import os
import torch
import torchvision
import json


# -----------------------------------------
# -- ref:
#   https://blog.csdn.net/qq_39507748/article/details/105394808
# -----------------------------------------


datasetdir = "../WasteClassification/dataset-garbage/images"
traindir = os.path.join(datasetdir, 'train')
valdir = os.path.join(datasetdir, 'val')

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(180),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize
])

print("Loading Data Set ...... ")
dataset = torchvision.datasets.ImageFolder(
    root=datasetdir,            # 图片存储的根目录，即各类别文件夹所在目录的上一级目录
    transform=transform,             # 对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片
    target_transform=None,
    is_valid_file=None          # 获取图像文件的路径并检查该文件是否为有效文件的函数(用于检查损坏文件)
)
print("Loading Data Set finished")
print(len(dataset))
print(dataset.class_to_idx)  # 按顺序为这些类别定义索引为0,1...


class_id={value:key for key,value in dataset.class_to_idx.items()}# class_to_idx是'battery': 0, 需要交互键值
info_json = json.dumps(
    class_id,
    sort_keys=False,
    indent=4,
    separators=(',', ': ')
)
# 显示数据类型
print(type(info_json))
f = open(datasetdir+'/garbage_class.json', 'w')
f.write(info_json)

