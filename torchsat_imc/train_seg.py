import os
import pathlib
import numpy as np
import gettext
_ = gettext.gettext

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from ignite.metrics import IoU, Precision, Recall # from pytorch-ignite

import imc_api
import dataset_utils
import torchsat_imc.transforms.transforms_seg as T_seg
from torchsat_imc.datasets.folder import SegmentationDataset
from torchsat_imc.models.utils import get_model
import torchsat_imc.imc_callbacks as imc_callbacks
# from torchsat.models.segmentation import unet_v2


#
# TODO: move losses to separate file, add choise of loss to training function
#







"""
Losses: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""

class DiceLoss(nn.Module):
    """
    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, labels, smooth=1):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        labels = labels.view(-1)
    
        intersection = (inputs * labels).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + labels.sum() + smooth)  
    
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    This loss combines Dice loss with the standard binary cross-entropy (BCE) loss that is generally the default for segmentation models. 
    Combining the two methods allows for some diversity in the loss, while benefitting from the stability of BCE. 
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
    
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
    
        return Dice_BCE


class IoULoss(nn.Module):
    """
    The IoU metric, or Jaccard Index, is similar to the Dice metric and is calculated as the ratio between the overlap 
    of the positive instances between two sets, and their mutual combined values
    """
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
    
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
    
        IoU = (intersection + smooth)/(union + smooth)
            
        return 1 - IoU


class FocalLoss(nn.Module):
    """
    Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as a means of combatting extremely imbalanced datasets 
    where positive cases were relatively rare. Their paper "Focal Loss for Dense Object Detection" is retrievable 
    here: https://arxiv.org/abs/1708.02002. In practice, the researchers used an alpha-modified version of the function 
    so I have included it in this implementation.
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
    
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                    
        return focal_loss


class TverskyLoss(nn.Module):
    """
    This loss was introduced in "Tversky loss function for image segmentation using 3D fully convolutional deep networks", 
    retrievable here: https://arxiv.org/abs/1706.05721. It was designed to optimise segmentation on imbalanced medical datasets 
    by utilising constants that can adjust how harshly different types of error are penalised in the loss function.

    To summarise, this loss function is weighted by the constants 'alpha' and 'beta' that penalise false positives and false negatives 
    respectively to a higher degree in the loss function as their value is increased. The beta constant in particular has applications 
    in situations where models can obtain misleadingly positive performance via highly conservative prediction. You may want to experiment 
    with different values to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
    
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
    
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    """
    A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
    
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
    
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                    
        return FocalTversky 


class TrainingMetrics:
    def __init__(self, loss: float = 0.0):
        self.loss = loss

class ValidationMetrics:
    def __init__(self, loss: float = 0.0, precision: float = 0.0, recall: float = 0.0, f1: float = 0.0):
        self.loss = loss
        self.precision = precision
        self.recall = recall
        self.f1 = f1


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device, writer) -> TrainingMetrics:
    print('train epoch {}'.format(epoch))

    model.train()
    softmax = nn.Softmax(dim=0)

    training_loss = 100.0

    for idx, data in enumerate(dataloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), (data[1].permute(0, 3, 1, 2).to(torch.float32).contiguous() / 255.0).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = softmax(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print('train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx+1, len(dataloader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(dataloader)*epoch+idx)
        training_loss = loss.item()
    
    return TrainingMetrics(loss)


def evaluation(epoch, dataloader, model, criterion, device, writer) -> ValidationMetrics:
    """
    Evaluation for onehot vector output segmentation
    """

    print('\neval epoch {}'.format(epoch))

    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []

    model.eval()
    softmax = nn.Softmax(dim=0)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), (data[1].permute(0, 3, 1, 2).to(torch.float32).contiguous() / 255.0).to(device)

            # calculate outputs by running images through the network
            outputs = model(inputs)
            outputs = softmax(outputs)
            loss = criterion(outputs, labels)

            preds_max = torch.round(outputs).to(torch.uint8)
            labels_max = torch.round(labels).to(torch.uint8)

            precision.update((preds_max, labels_max))
            recall.update((preds_max, labels_max))
            mean_loss.append(loss.item())
            mean_recall.append(recall.compute().item())
            mean_precision.append(precision.compute().item())

            # print('val-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx + 1, len(dataloader), loss.item()))
            writer.add_scalar('test/loss', loss.item(), len(dataloader) * epoch + idx)

    mean_loss_value = np.array(mean_loss).mean()
    mean_precision, mean_recall = np.array(mean_precision).mean(), np.array(mean_recall).mean()
    f1 = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-20)

    print('precision: {:07.5}, recall: {:07.5}, f1: {:07.5}\n'.format(mean_precision, mean_recall, f1))
    writer.add_scalar('test/epoch-loss', mean_loss_value, epoch)
    writer.add_scalar('test/f1', f1, epoch)
    writer.add_scalar('test/precision', mean_precision, epoch)
    writer.add_scalar('test/recall', mean_recall, epoch)

    return ValidationMetrics(mean_loss_value, mean_precision, mean_recall, f1)



def load_data(traindir, valdir, **kwargs):
    """generate the train and val dataloader, you can change this for your specific task

    Args:
        traindir (str): train dataset dir
        valdir (str): validation dataset dir

    Returns:
        tuple: the train dataset and validation dataset
    """

    # TODO: compose from params

    train_transform = T_seg.Compose([
        T_seg.RandomCrop(int(kwargs['crop_size'])),
        T_seg.RandomHorizontalFlip(),
        T_seg.RandomVerticalFlip(),
        T_seg.ToTensor(),
        T_seg.Normalize(kwargs['mean'], kwargs['std']),
    ])

    val_transform = T_seg.Compose([
        T_seg.ToTensor(),
        T_seg.Normalize(kwargs['mean'], kwargs['std']),
    ])

    dataset_train = SegmentationDataset(traindir, image_extensions=kwargs['image_extensions'], label_extension=kwargs['label_extension'], transforms=train_transform)
    dataset_val = SegmentationDataset(valdir, image_extensions=kwargs['image_extensions'], label_extension=kwargs['label_extension'], transforms=val_transform)

    return dataset_train, dataset_val


def train(training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr, 
          train_path: pathlib.Path, val_path: pathlib.Path, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225], 
          image_extensions: list = ('jpg'), label_extension: str = 'png', model: str = 'unet34', pretrained: bool = True,
          resume: pathlib.Path = "", num_input_channels: int = 3, num_output_classes: int = 3, crop_size: int = 512, device: str = 'cpu', 
          batch_size: int = 16, epochs: int = 90, lr: float = 0.001, print_freq: int = 10, ckp_dir: pathlib.Path = './') -> bool:
    """Training segmentation model
    
    Args:
        train_path (str): train dataset path
        val_path (str): validate dataset path
        mean (list): dataset mean
        std (list): dataset std 
        image_extensions (list): list of image extensions
        label_extension (str): label extension
        model (str): model name
        pretrained (bool): load model pretrained weights
        resume (str): path to the latest checkpoint
        num_input_channels (int): input image channels
        num_output_classes (int): num of classes in the output
        crop_size (int): random crop size
        device (str): 'cpu' of 'gpu', device to train model on
        batch_size (int): training batch size
        epochs (int): number of training epochs
        lr (float): initial training learning rate
        print_freq (int): metric values print frequency
        ckp_dir (str): path to save checkpoint
    """

    result = True

    try:

        imc_callbacks.confirm_running(training_panel)

        torch.backends.cudnn.benchmark = False
        if device == 'cuda' and not torch.cuda.is_available():
            imc_callbacks.show_message(imc_api.MessageTitle.LogInfo, _("CUDA is not available. Falling back to CPU"))   
            device = 'cpu'
        
        device = torch.device('cuda' if device == 'cuda' else 'cpu')
        torch.cuda.empty_cache()

        if len(mean) != len(std):
            imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("The standart deviation array must be the same size as the mean array"))        
            imc_callbacks.stop_training(training_panel)
            return False

        if len(mean) != num_input_channels:
            imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("Number of input channels must be the same as the size of mean and std arrays"))        
            imc_callbacks.stop_training(training_panel)
            return False

        # dataset and dataloader
        train_data, val_data = load_data(train_path, val_path, image_extensions=image_extensions, label_extension=label_extension, crop_size=crop_size, mean=mean, std=std)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        # model
        model = get_model(model, num_input_channels, pretrained=pretrained)
        model.to(device)
        if resume:
            model.load_state_dict(torch.load(resume, map_location=device))

        # loss
        # criterion = nn.BCELoss()
        # criterion = DiceLoss()
        # criterion = DiceBCELoss()
        criterion = FocalLoss()

        # optim and lr scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        writer = SummaryWriter(ckp_dir)
        for epoch in range(epochs):
            learning_rate = lr_scheduler.get_lr()[0]
            writer.add_scalar('train/lr', learning_rate, epoch)
            
            # train
            training_metrics = train_one_epoch(epoch, train_loader, model, criterion, optimizer, device, writer)
            validation_metrics = evaluation(epoch, val_loader, model, criterion, device, writer)
            lr_scheduler.step()

            # save checkpoint 
            checkpoint_name = f"cls_epoch_{epoch}"
            training_checkpoint = imc_api.TrainingCheckpoint(checkpoint_name, learning_rate, training_metrics.loss, validation_metrics.loss, 
                                                             validation_metrics.precision, validation_metrics.recall, validation_metrics.f1)
            imc_callbacks.update_epoch(epoch, training_panel)
            imc_callbacks.add_checkpoint(training_checkpoint, training_panel)
            torch.save(model.state_dict(), os.path.join(ckp_dir, checkpoint_name + ".pth"))
        
        result = True
            
    except Exception as e:
        imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("Error occured during training"), str(e))        
        result = False
    finally:
        imc_callbacks.stop_training(training_panel)
        return result


def train_segmentation(params: imc_api.TrainingParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> bool:
    """Training segmentation model
    
    Args:
        params (imc_api.TrainingParams): training params
        training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks
        progress_bar (imc_api.ProgressBarPtr): progress bar ptr for progress updates

    """

    tiles_extension = "tif"

    # 1. Split dataset into training and validation images and rasterize labels
    train_path, val_path = dataset_utils.split_dataset(features_path = params.features_path, 
                                                       labels_path = params.labels_path, 
                                                       dataset_split = params.dataset_split,
                                                       tile_size = params.crop_size,
                                                       tiles_extension = tiles_extension)

    # 2. Train on splitted dataset
    result = train(
          training_panel = training_panel,
          progress_bar = progress_bar,
          train_path = train_path, 
          val_path = val_path, 
          mean = params.mean, 
          std = params.std,
          image_extensions = params.image_extensions, 
          label_extension = params.lable_extension,
          use_gaussian_blur = params.use_gaussian_blur, gaussian_blur_kernel_size = params.gaussian_blur_kernel_size,
          use_noise = params.use_noise, noise_type = params.noise_type, noise_percent = params.noise_percent,
          use_brightness = params.use_brightness, brightness_max_value = params.brightness_max_value,
          use_contrast = params.use_contrast, contrast_max_value = params.contrast_max_value,
          use_shift = params.use_shift, shift_max_percent = params.shift_max_percent,
          use_rotation = params.use_rotation, rotation_max_left_angle_value = params.rotation_max_left_angle_value, rotation_max_right_angle_value = params.rotation_max_right_angle_value,
          use_horizontal_flip = params.use_horizontal_flip, horizontal_flip_probability = params.horizontal_flip_probability, 
          use_vertical_flip = params.use_vertical_flip, vertical_flip_probability = params.vertical_flip_probability,
          use_flip = params.use_flip, flip_probability = params.flip_probability, 
          crop_size = params.crop_size,
          model = params.model_name,
          pretrained = params.pretrained,
          resume = params.resume_path, 
          num_input_channels = params.num_input_channels,
          num_output_classes = params.num_output_classes,
          crop_size = params.crop_size,
          device = params.device,
          batch_size = params.batch_size,
          epochs = params.epochs, 
          lr = params.learning_rate, 
          print_freq = params.print_freq, 
          ckp_dir = params.ckp_dir)

    return result
