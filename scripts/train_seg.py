import argparse
import os

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import IoU, Precision, Recall # from pytorch-ignite

import torchsat.transforms.transforms_seg as T_seg
from torchsat.datasets.folder import SegmentationDataset
from torchsat.models.utils import get_model
from torchsat.models.segmentation import unet_v2


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device, writer):
    print('train epoch {}'.format(epoch))
    model.train()

    softmax = nn.Softmax(dim=0)

    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # inp = torch.randn(3, 5, requires_grad=True)
        # tar = torch.empty(3, dtype=torch.long).random_(5)
        # out = criterion(inp, tar)

        targets = targets.permute(0, 3, 1, 2).to(torch.float32).contiguous()
        loss = criterion(softmax(outputs), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx+1, len(dataloader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(dataloader)*epoch+idx)


def evalidation(epoch, dataloader, model, criterion, device, writer):
    print('\neval epoch {}'.format(epoch))
    model.eval()
    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            
            targets = targets.permute(0, 3, 1, 2).to(torch.float32).contiguous()
            loss = criterion(outputs, targets)

            preds_max = outputs.argmax(1)
            targets_max = targets.argmax(1)

            precision.update((preds_max, targets_max))
            recall.update((preds_max, targets_max))
            mean_loss.append(loss.item())
            mean_recall.append(recall.compute().item())
            mean_precision.append(precision.compute().item())

            # print('val-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx + 1, len(dataloader), loss.item()))
            writer.add_scalar('test/loss', loss.item(), len(dataloader) * epoch + idx)

    mean_precision, mean_recall = np.array(mean_precision).mean(), np.array(mean_recall).mean()
    f1 = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-20)

    print('precision: {:07.5}, recall: {:07.5}, f1: {:07.5}\n'.format(mean_precision, mean_recall, f1))
    writer.add_scalar('test/epoch-loss', np.array(mean_loss).mean(), epoch)
    writer.add_scalar('test/f1', f1, epoch)
    writer.add_scalar('test/precision', mean_precision, epoch)
    writer.add_scalar('test/recall', mean_recall, epoch)


def load_data(traindir, valdir, **kwargs):
    """generate the train and val dataloader, you can change this for your specific task

    Args:
        traindir (str): train dataset dir
        valdir (str): validation dataset dir

    Returns:
        tuple: the train dataset and validation dataset
    """

    train_transform = T_seg.Compose([
        T_seg.RandomCrop(int(kwargs['crop_size'])),
        T_seg.RandomHorizontalFlip(),
        T_seg.RandomVerticalFlip(),
        T_seg.ToTensor(),
        T_seg.Normalize(),
    ])
    val_transform = T_seg.Compose([
        T_seg.ToTensor(),
        T_seg.Normalize(),
    ])

    print(kwargs['image_extensions'])
    print(repr(SegmentationDataset))

    dataset_train = SegmentationDataset(traindir, image_extensions=kwargs['image_extensions'], label_extension=kwargs['label_extension'], transforms=train_transform, )
    dataset_val = SegmentationDataset(valdir, image_extensions=kwargs['image_extensions'], label_extension=kwargs['label_extension'], transforms=val_transform)

    return dataset_train, dataset_val



import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')

    # dataset and dataloader
    train_data, val_data = load_data(args.train_path, args.val_path, image_extensions=args.image_extensions, label_extension=args.label_extension, crop_size=args.crop_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    # model
    model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    # model = unet_v2.UNet50()
    model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # TODO: resume learning rate

    # loss
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.BCELoss().to(device)
    # criterion = DiceLoss()

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter(args.ckp_dir)
    for epoch in range(args.epochs):
        writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)
        evalidation(epoch, val_loader, model, criterion, device, writer)
        train_one_epoch(epoch, train_loader, model, criterion, optimizer, device, writer)
        evalidation(epoch, val_loader, model, criterion, device, writer)
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.ckp_dir, 'cls_epoch_{}.pth'.format(epoch)))


def parse_args():
    parser = argparse.ArgumentParser(description='TorchSat Segmentation Training Script')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--val-path', help='validate dataset path')
    parser.add_argument('--image_extensions', nargs='+', default='jpg', help='image extension')
    parser.add_argument('--label_extension', default='png', help='label extension')
    parser.add_argument('--model', default="unet34", help='')
    parser.add_argument('--pretrained', default=True)

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=3, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')
    parser.add_argument('--crop-size', default=512, type=int, help='random crop size')

    parser.add_argument('--device', default='cpu')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='./', help='path to save checkpoint')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
