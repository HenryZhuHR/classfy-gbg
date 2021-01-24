
# ref:https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
from torchvision import transforms
from PIL import Image
import urllib
import torch
import torchvision


def load_model(args):
    """
        加载模型
        ---
        输出全连接层做如下优化:
        ```python
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, args.num_classes)
        )
        ```
    """
    # 32*4d表示32个paths，每个path的宽度为4
        # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
        # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        # model=torchvision.models.resnext50_32x4d()
    print("=> creating model '{}'".format(args.arch))
    model = torchvision.models.__dict__[args.arch](progress=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),    
        torch.nn.Linear(2048, args.num_classes)
    )
    return model

def inference():
    filename="./dog.jpg"
    model=load_model()

    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    print(torch.nn.functional.softmax(output[0], dim=0))

    print(torch.min(torch.nn.functional.softmax(output[0], dim=0)))

"""### Model Description
The provided ResNeXt models are pre-trained in weakly-supervised fashion on **940 million** public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning on ImageNet1K dataset.  Please refer to "Exploring the Limits of Weakly Supervised Pretraining" (https://arxiv.org/abs/1805.00932) presented at ECCV 2018 for the details of model training.

We are providing 4 models with different capacities.

| Model              | #Parameters | FLOPS | Top-1 Acc. | Top-5 Acc. |
| ------------------ | :---------: | :---: | :--------: | :--------: |
| ResNeXt-101 32x8d  | 88M         | 16B   |    82.2    |  96.4      |
| ResNeXt-101 32x16d | 193M        | 36B   |    84.2    |  97.2      |
| ResNeXt-101 32x32d | 466M        | 87B   |    85.1    |  97.5      |
| ResNeXt-101 32x48d | 829M        | 153B  |    85.4    |  97.6      |

Our models significantly improve the training accuracy on ImageNet compared to training from scratch. **We achieve state-of-the-art accuracy of 85.4% on ImageNet with our ResNext-101 32x48d model.**

### References

 - [Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932)
"""

