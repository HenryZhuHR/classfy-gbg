#!C:\Users\henryzhu\Programs\Python\Python38\python.exe
#-*- coding:utf-8 _*-
import os, time
import json
import torch
import torchvision
from PIL import Image
from predict.resnetxt_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl

# 选择可以使用的 显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 单卡使用 仅单卡GPU可见
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 多卡使用

# 320*320
args = {
    'arch': 'resnext101_32x16d_wsl',
    'pretrained': False,
    'num_classes': 43,
    'image_size': 320
}
import pprint
pprint.pprint(args)
print()

# data/garbage_classify_v2/garbage_classify_rule.json


class ClassficationService():
    def __init__(self, model_path):
        self.model = self.load_model(model_path)  # 加载模型
        self.pre_img = self.preprocess_img()  # 图片预处理
        self.model.eval()
        self.device = torch.device(
            'cuda' if not torch.cuda.is_available() else 'cpu')
        with open('checkepoints/garbage_class.json','r',encoding='utf8')as fp:
            self.label_id_name_dict = json.load(fp)
        print(self.label_id_name_dict)

    def load_model(self, model_path):
        """加载模型"""
        print(" -- Loading model")
        model = None
        # 加载模型
        if args['arch'] == 'resnext101_32x16d_wsl':
            model = resnext101_32x16d_wsl(pretrained=False, progress=False)
        elif args['arch'] == 'efficientnet-b7':
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_name(args['arch'])
        else:
            model = torchvision.models.alexnet()
        # pprint.pprint(model)    # 打印模型结构

        # 修改最后一层的参数, 修改输出的全连接层的类别数量为 args['num_classes']
        layerName, layer = list(model.named_children())[-1]  # 获取最后一层
        # print(layerName)    # fc
        # print(layer)        # Linear(in_features=2048, out_features=1000, bias=True)
        exec_str = "model." + layerName + "=torch.nn.Linear(layer.in_features," + str(
            args['num_classes']) + ")"
        exec(exec_str)

        # 选择推理模型的架构 CPU/GPU
        if not torch.cuda.is_available():
            print("   -- Inference using GPU")
            modelState = torch.load(model_path)
            if torch.cuda.device_count(
            ) > 1:  # 如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"。
                model = torch.nn.DataParallel(model)
            model.load_state_dict(modelState,
                                  strict=False)  # strict=False 避免出错
            model = model.cuda()
        else:
            print("   -- Inference using CPU")
            modelState = torch.load(model_path, map_location='cpu')
            model.load_state_dict(modelState, strict=False)
        print(" -- Finished to load model")
        return model

    def preprocess_img(self):
        """
            图像的预处理
            torchvision: https://zhuanlan.zhihu.com/p/58511839?from_voters_page=true
        """
        infer_transformation = torchvision.transforms.Compose([
            Resize((args['image_size'], args['image_size'])),  # 尺寸剪裁 (resize)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(   # 归一化 (normalize)
                mean=[0.485, 0.456, 0.406],     # 均值
                std=[0.229, 0.224, 0.225]       # 标准差
            ),
        ])
        return infer_transformation

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.pre_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        ---
        模型推理函数
        ---
        输入
        {'input_img': img}

        Here are a inference example of resnet, if you use another model, please modify this function
        """
        
        img = data['input_img']
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            pred_score = self.model(img)

        print("   -- predict score",pred_score)
        print("   -- predict score output length",len(pred_score[0]))
        
        if pred_score is not None:
            _, pred_label = torch.max(pred_score.data, 1)
            print("str(pred_label[0].item()): ",str(pred_label[0].item()))
            result = {
                'result': self.label_id_name_dict[str(pred_label[0].item())]            }
        else:
            result = {'result': 'predict score is None'}

        return result

    def _postprocess(self, data):
        return data


class Resize(object):
    """改变图片尺寸"""
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))
        img = img.resize(self.size, self.interpolation)
        return img


if __name__ == '__main__':
    # 训练好的模型路径
    model_path = "checkepoints/model_27_9942_7042.pth"
    # 输入的测试图片路径
    input_dir = "data/garbage_classify_v2/train_data_v2"
    input_dir = "images"
    

    infer = ClassficationService(model_path=model_path)
    files = os.listdir(input_dir)
    print(files)
    print(" -- Test images count: ",len(files))

    # Start inference
    t1 = time.time()
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        # print(file_path,input_dir,file_name)
        img = Image.open(file_path)
        # print(img)

        img = infer.pre_img(img)
        print(" -- start inference")
        tt1 = time.time()
        result = infer._inference({'input_img': img})
        print("    -- label/result: {}/{}".format(file_name,result["result"]))
        tt2 = time.time()
        print("   -- current inference time %3f s"%(tt2 - tt1))
    t2 = time.time()
    print(" -- Total Inference Time:", "%3f s"%(t2 - t1) )

