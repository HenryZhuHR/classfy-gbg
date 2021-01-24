import torch
import torchvision

# Models
default_model_names = sorted(name for name in torchvision.models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(torchvision.models.__dict__[name]))

customized_models_names = sorted(name for name in torchvision.models .__dict__
                                 if not name.startswith("__")
                                 and callable(torchvision.models .__dict__[name]))

for name in torchvision.models .__dict__:
    if not name.startswith("__") and callable(torchvision.models .__dict__[name]):
        torchvision.models.__dict__[name] = torchvision.models .__dict__[name]

model_names = default_model_names + customized_models_names


def make_model(args):
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
    print("=> creating model '{}'".format(args.arch))
    import pprint
    pprint.pprint(torchvision.models.__dict__)
    model = torchvision.models.__dict__[args.arch](progress=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),    
        torch.nn.Linear(2048, args.num_classes)
    )
    return model


if __name__=='__main__':
    all_model = sorted(name for name in torchvision.models.__dict__ if not name.startswith("__"))
    print("==> all_model:")
    print(all_model)
