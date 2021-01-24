from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms

from models.build_net import model_names
from models.build_net import make_model
import models

from utils import dataset
from utils.transform import get_transforms

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-train',
                    default='data/new_shu_label.txt',
                    type=str,
                    help="训练集的地址")  # 训练集txt文件
parser.add_argument('-val', default='data/val1.txt', type=str,
                    help="验证集的地址")  # 验证集txt文件

parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')  # 总训练轮数
parser.add_argument('--num-classes',
                    default=43,
                    type=int,
                    metavar='N',
                    help='number of classfication of image')  # 数据集类别数量
parser.add_argument('--image-size',
                    default=288,
                    type=int,
                    metavar='N',
                    help='the train image size')  # 图像尺寸
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number')  # 开始训练的epoch
parser.add_argument('--train-batch',
                    default=1,
                    type=int,
                    metavar='N',
                    help='train batchsize (default: 256)')  # 训练时的 batch size
parser.add_argument('--test-batch',
                    default=1,
                    type=int,
                    metavar='N',
                    help='test batchsize (default: 200)')  # 验证时的 batch size
parser.add_argument('--optimizer',
                    default='sgd',
                    choices=['sgd', 'rmsprop', 'adam', 'AdaBound', 'radam'],
                    metavar='N',
                    help='optimizer (default=sgd)')  # 训练优化器选择
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate，1e-2， 1e-4, 0.001')  # 学习率
parser.add_argument('--lr-fc-times',
                    '--lft',
                    default=5,
                    type=int,
                    metavar='LR',
                    help='initial model last layer rate')
parser.add_argument('--drop',
                    '--dropout',
                    default=0,
                    type=float,
                    metavar='Dropout',
                    help='Dropout ratio')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[30, 50, 60],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma',
                    type=float,
                    default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov',
                    dest='nesterov',
                    action='store_false',
                    help='do not use Nesterov momentum')
parser.add_argument('--alpha',
                    default=0.99,
                    type=float,
                    metavar='M',
                    help='alpha for ')
parser.add_argument('--beta1',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2',
                    default=0.999,
                    type=float,
                    metavar='M',
                    help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--final-lr',
                    '--fl',
                    default=1e-3,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-3)')
# Checkpoints
parser.add_argument('--checkpoint',
                    default='./checkepoints',
                    type=str,
                    metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnext101_32x16d_wsl',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext101_32x8d, pnasnet5large)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality',
                    type=int,
                    default=32,
                    help='ResNet cardinality (group).')
parser.add_argument('--base-width',
                    type=int,
                    default=4,
                    help='ResNet base width.')
parser.add_argument('--widen-factor',
                    type=int,
                    default=4,
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id',
                    default='0',
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    print(" -- Train with GPU")
    torch.cuda.manual_seed_all(args.manualSeed)
else:
    print(" -- Train with CPU")
best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # 加载数据集
    print('==> data augmentation...')
    transform = get_transforms(input_size=args.image_size,
                               test_size=args.image_size,
                               backbone=None)  # 数据增强

    print('==> Preparing dataset %s ...' % args.train)
    trainset = dataset.Dataset(root=args.train,
                               transform=transform['val_train'])
    train_loader = data.DataLoader(trainset,
                                   batch_size=args.train_batch,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   pin_memory=True)
    valset = dataset.TestDataset(root=args.val,
                                 transform=transform['val_test'])
    val_loader = data.DataLoader(valset,
                                 batch_size=args.test_batch,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 pin_memory=True)
    # 加载模型,并且做并行化处理
    model = models.model.load_model(args)
    # model=make_model(args)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print(' ==> Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=5,
                                                           verbose=False)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'),
                        title=title,
                        resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names([
            'LearningRate', 'Loss', 'Valid Loss', 'Train Acc.',
            'Valid Acc.'
        ])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch,
                                   use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\n Epoch: %d/%d lr: %f' %
              (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc, train_5 = train(train_loader, model, criterion,
                                               optimizer, epoch, use_cuda)
        test_loss, test_acc, test_5 = test(val_loader, model, criterion, epoch,
                                           use_cuda)

        scheduler.step(test_loss)

        # append logger file
        logger.append(
            [state['lr'], train_loss, test_loss, train_acc, test_acc])
        print(
            'train_loss:%f, val_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f'
            % (train_loss, test_loss, train_acc, train_5, test_acc, test_5))

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if len(args.gpu_id) > 1:
            save_checkpoint(
                {
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'train_acc': train_acc,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                single=True,
                checkpoint=args.checkpoint)
        else:
            save_checkpoint(
                {
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_acc': train_acc,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                single=True,
                checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # 训练时，实时输出是训练信息
        bar.suffix = '({batch}/{size}) -Data: {data:.3f}s -Batch: {bt:.3f}s -Total: {total:} -ETA: {eta:} -Loss: {loss:.3f} -top1: {top1: .3f} -top5: {top5: .3f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)


if __name__ == '__main__':
    main()
