import os

space = ' '
os.system("python -V")

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

cmd = 'python src/main.py'+space

cmd += './data'+space     # 数据集目录
                        # images-folder with train and val folders

cmd += '--epochs'+space+'500'+space         # 训练轮数 epoch (default=100)
cmd += '--start-epoch'+space+'0'+space    # 开始训练的轮数，默认从头开始，继续训练
                                            #   记住上一次的训练结束的 epoch  (default=0)
cmd += '--batch-size'+space+'2'+space       # 送入网络的 batch size  (default=2)

# cmd += '--pretrained'+space+'2'+space       # 送入网络的 batch size  (default=2)
cmd += '--gpu'+space+'0'+space       # GPU 使用情况



cmd += '--learning-rate'+space+'0.1'+space      # 超参数，学习率 (default=0.1)
cmd += '--print-freq'+space+'100'+space         # 打印间隔


# cmd+=">checkpoints/train.log"                                 
print()
print(" ------ command ------ ")
print(cmd)
print()
os.system(cmd)    # 执行


