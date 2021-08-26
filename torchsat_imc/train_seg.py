import os
import json
from pathlib import Path
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



def load_data(features_dirpath: Path, labels_dirpath: Path, train_item_filenames: set, val_item_filenames: set, **kwargs):
    """generate the train and val dataloader, you can change this for your specific task

    Args:
        features_dirpath (pathlib.Path): path to directory with features
        labels_dirpath (pathlib.Path): path to directory with labels
        train_item_filenames (set): filenames of training images to load from feature and label directories
        val_item_filenames (set): filenames of validation images to load from feature and label directories

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

    dataset_train = SegmentationDataset(features_dirpath, labels_dirpath, train_item_filenames, transforms=train_transform)
    dataset_val = SegmentationDataset(features_dirpath, labels_dirpath, val_item_filenames, transforms=val_transform)

    return dataset_train, dataset_val


def train(training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr, 
          features_dirpath: Path, labels_dirpath: Path, 
          train_item_filenames: set, val_item_filenames: set, 
          mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225], 
          image_extensions: list = ('jpg'), label_extension: str = 'png', model: str = 'unet34', pretrained: bool = True,
          resume: Path = "", num_input_channels: int = 3, num_output_classes: int = 3, crop_size: int = 512, device: str = 'cpu', 
          batch_size: int = 16, epochs: int = 90, lr: float = 0.001, print_freq: int = 10, ckp_dir: Path = './') -> bool:
    """Training segmentation model
    
    Args:
        features_dirpath (pathlib.Path): path to directory with features
        labels_dirpath (pathlib.Path): path to directory with labels
        train_item_filenames (set): filenames of training images to load from feature and label directories
        val_item_filenames (set): filenames of validation images to load from feature and label directories
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
        train_data, val_data = load_data(features_dirpath, labels_dirpath, train_item_filenames, val_item_filenames, crop_size=crop_size, mean=mean, std=std)
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
    
    # extension of tiles
    tiles_extension = ".tif"
    rasterized_label_extension = ".tif"
    id_separator = "__"

    # changes to the dataset since last run
    item_num_changed = False      # number of items changed
    class_num_changed = False     # number of classes changed
    crop_size_changed = False     # crop size parameter changed
    has_modified_features = False # some items were modified
    has_modified_labels = False   # some labels were modified

    modified_features_ids = set() # item ids with modified features
    modified_labels_ids = set()   # item ids with modified labels    
    new_item_ids = set()          # added items
    new_classes_ids = set()       # added classes
    deleted_item_ids = set()      # deleted items
    deleted_classes_ids = set()   # deleted classes
    
    splitted_dataset_path = params.features_path.parent / "dataset"
    rasterized_labels_path = splitted_dataset_path / "rasterized"
    features_path = splitted_dataset_path / "features"
    labels_path = splitted_dataset_path / "labels"

    train_config_path = splitted_dataset_path / "config.json"

    # create directories if don't exist

    for path in [splitted_dataset_path, rasterized_labels_path, features_path, labels_path]:
        if not path.exists():
            path.mkdir()

    dataset_item_ids = set(os.listdir(params.labels_path)) # notice: dir names don't have extensions 

    # get params from last training config

    if not train_config_path.exists():
        train_config_path.touch()
    else:

        # read params of the last run
        f = open(train_config_path)
        config_json = json.load(f)
        previous_crop_size = config_json["crop_size"]
        previous_class_list = set(config_json["classes"])
        previous_dataset_items = config_json["items"] # dictionary ['item id': ('feature last update date', 'label last update date')]    

        # compare params of the last run with current

        # crop size
        if previous_crop_size != params.crop_size:
            crop_size_changed = True

        # classes
        classes_list = set(os.listdir(params.labels_path / dataset_item_ids[0]))
        new_classes_ids = classes_list - previous_class_list                      # classes added since last run
        deleted_classes_ids = previous_class_list - classes_list                  # classes deleted since last run
        if len(deleted_classes_ids) != 0 or len(new_classes_ids) != 0:
            class_num_changed = True

        # items
        new_item_ids = dataset_item_ids - set(previous_dataset_items.keys())      # item ids added since last run
        deleted_item_ids = set(previous_dataset_items.keys()) - dataset_item_ids  # item ids deleted since last run
        if len(deleted_item_ids) != 0 or len(new_item_ids) != 0:
            item_num_changed = True

        common_item_ids = dataset_item_ids.intersection(set(previous_dataset_items.keys()))

        for feature_filename in os.listdir(params.features_path):
            # if feature modified since last run
            item_id = Path(feature_filename).with_suffix('')
            previous_update_date = previous_dataset_items.get(item_id)
            if previous_update_date == None:
                continue
            # if prev run update date is earlier - add to modified list 
            stat = os.stat(params.features_path / feature_filename)
            if stat.st_mtime > previous_update_date[0]:
                modified_features_ids.add(item_id)                                 # features modified since last run
                has_modified_features = True

        for item_id in common_item_ids:
            # if label modified since last run
            previous_update_date = previous_dataset_items.get(item_id)
            if previous_update_date == None:
                continue
            label_dirpath = params.labels_path / item_id
            for label in os.listdir(label_dirpath):
                stat = os.stat(label_dirpath / label)
                # if at least one class modified - add to modified list 
                if previous_update_date[1] < stat.st_mtime:
                    modified_labels_ids.add(item_id)                               # labels modified since last run
                    has_modified_labels = True
                    break
    #
    # 1. Rasterize labels
    #

    if class_num_changed:
        # remove all rasterized labels
        for child in rasterized_labels_path.iterdir():
            if child.is_file():
                child.unlink(True)
        # rasterize all labels
        dataset_utils.rasterize_labels( item_ids=dataset_item_ids, 
                                        labels_path=params.labels_path, 
                                        out_dir=rasterized_labels_path,
                                        out_ext=rasterized_label_extension)
    elif item_num_changed or has_modified_labels:
        # remove rasterized labels of deleted and modified items
        for id_list in [deleted_item_ids, modified_labels_ids]:
            for id in id_list:
                ((rasterized_labels_path / id).with_suffix(rasterized_label_extension)).unlink(True)
        # rasterize new labels
        dataset_utils.rasterize_labels( item_ids=new_item_ids,
                                        labels_path=params.labels_path, 
                                        out_dir=rasterized_labels_path,
                                        out_ext=rasterized_label_extension)
        # rasterize modified labels
        dataset_utils.rasterize_labels( item_ids=modified_labels_ids, 
                                        labels_path=params.labels_path, 
                                        out_dir=rasterized_labels_path,
                                        out_ext=rasterized_label_extension)
    else:
        pass
    
    #
    # 2. Split rasterized labels on tiles
    #

    if class_num_changed or crop_size_changed:
        # remove all splitted labels
        for child in labels_path.iterdir():
            if child.is_file():
                child.unlink(True)
        # split all labels
        dataset_utils.split_images( images_path = rasterized_labels_path, 
                                    item_ids = dataset_item_ids,
                                    crop_size = params.crop_size, 
                                    id_separator = id_separator,
                                    out_dir = labels_path,
                                    out_ext = tiles_extension)
        
    elif item_num_changed or has_modified_labels:
        # remove splitted labels of deleted and modified items
        for id_list in [deleted_item_ids, modified_labels_ids]:
            for id in id_list:
                ((rasterized_labels_path / id).with_suffix(tiles_extension)).unlink(True)
        # split new labels
        dataset_utils.split_images( images_path = rasterized_labels_path, 
                                    item_ids = new_item_ids, 
                                    crop_size = params.crop_size, 
                                    id_separator = id_separator,
                                    out_dir = labels_path,
                                    out_ext = tiles_extension)
        # split modified labels
        dataset_utils.split_images( images_path = rasterized_labels_path, 
                                    item_ids = modified_labels_ids, 
                                    crop_size = params.crop_size, 
                                    id_separator = id_separator,
                                    out_dir = labels_path,
                                    out_ext = tiles_extension)
    else:
        pass
    
    #
    # 3. Split features on tiles
    #

    if crop_size_changed:
        # remove all features
        for child in labels_path.iterdir():
            if child.is_file():
                child.unlink(True)
        # split all features
        dataset_utils.split_images( images_path = params.features_path, 
                                    item_ids = dataset_item_ids, 
                                    crop_size = params.crop_size, 
                                    id_separator = id_separator,
                                    out_dir = features_path,
                                    out_ext = tiles_extension)
    elif item_num_changed or has_modified_features:
        # remove features of deleted and modified items
        for id_list in [deleted_item_ids, modified_features_ids]:
            for id in id_list:
                ((features_path / id).with_suffix(tiles_extension)).unlink(True)
        # split new features
        dataset_utils.split_images( images_path = params.features_path, 
                                    item_ids = new_item_ids, 
                                    crop_size = params.crop_size, 
                                    id_separator = id_separator,
                                    out_dir = features_path,
                                    out_ext = tiles_extension)
        # split modified features
        dataset_utils.split_images( images_path = params.features_path, 
                                    item_ids = modified_features_ids, 
                                    crop_size = params.crop_size, 
                                    id_separator = id_separator,
                                    out_dir = features_path,
                                    out_ext = tiles_extension)
    else:
        pass


    # check that number of feature tiles is the same as number of label tiles

    ext_len = len(tiles_extension)
    splitted_dataset_item_ids = []
    splitted_features_ids_list = set([filename[:-ext_len] for filename in os.listdir(features_path)])
    splitted_labels_ids_list = set([filename[:-ext_len] for filename in os.listdir(labels_path)])
    if len(splitted_features_ids_list) != len(splitted_labels_ids_list):
        imc_callbacks.log_message(imc_api.MessageTitle.LogInfo, f"Number of feature tiles ({len(splitted_features_ids_list)}) is not the same as number of label tiles ({len(splitted_labels_ids_list)})! Trying to fix...")

        # try to take only features and labels with intersecting names

        intersecting_item_ids = splitted_features_ids_list.intersect(splitted_labels_ids_list)
        if len(intersecting_item_ids) <= 0:
            imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("No valid items in the dataset"))
            imc_callbacks.stop_training(training_panel)
            return False
        
        splitted_dataset_item_ids = intersecting_item_ids

        # delete all non-intersecting names

        unique_feature_ids = (splitted_features_ids_list - splitted_labels_ids_list)
        for id in unique_feature_ids:
            (features_path / id).with_suffix(tiles_extension).unlink(True)

        unique_label_ids = (splitted_labels_ids_list - splitted_features_ids_list)
        for id in unique_label_ids:
            (labels_path / id).with_suffix(tiles_extension).unlink(True)
    else:
        splitted_dataset_item_ids = splitted_labels_ids_list

    # update train config
    
    dataset_items = dict()

    for id in dataset_item_ids:
        
        feature_filename = ""
        feature_found = False
        for filename in os.listdir(params.features_path):
            if Path(filename).with_suffix('') == id:
                feature_filename = filename
                break
        
        label_dir = params.labels_path / id

        if feature_found and label_dir.exists():
            # find feature latest update time
            feature_path = params.features_path / feature_filename
            stat = os.stat(feature_path)
            feature_mtime = stat.st_mtime

            # find label latest update time
            label_mtime = 0 # find the latest modif. time to save as modif. time of the label
            for class_filename in os.listdir(label_dir):
                label_full_path = label_dir / class_filename
                stat = os.stat(label_full_path)
                if label_mtime < stat.st_mtime:
                    label_mtime = stat.st_mtime
            
            dataset_items.update({id: (feature_mtime, label_mtime)})

    config_json = {}
    config_json["crop_size"] = params.crop_size
    config_json["classes"] = classes_list
    config_json["items"] = dataset_items
    with open(train_config_path, 'w') as outfile:
        json.dump(config_json, outfile)

    # split dataset items on training and validation

    dataset_split_idx = int(len(splitted_dataset_item_ids) * params.dataset_split)
    
    train_dataset_item_ids = set(splitted_dataset_item_ids[:dataset_split_idx])
    if len(train_dataset_item_ids) <= 0:
        imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("No images in the dataset for training. Try adding some"))
        imc_callbacks.stop_training(training_panel)
        return False

    val_dataset_item_ids = set(splitted_dataset_item_ids[dataset_split_idx:])
    if len(val_dataset_item_ids) <= 0:
        imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("No images in the dataset for validation. Try adding more or changing dataset split rate"))
        imc_callbacks.stop_training(training_panel)
        return False

    imc_callbacks.log_message(imc_api.MessageTitle.LogInfo, f"Using {len(train_dataset_item_ids)} images for training, {len(train_dataset_item_ids)} images for validation")

    # 
    # 4. Train on splitted dataset
    #
    
    result = train(
          training_panel = training_panel,
          progress_bar = progress_bar,
          features_dirpath = features_path, 
          labels_dirpath = labels_path,
          train_item_filenames = train_dataset_item_ids,
          val_item_filenames = val_dataset_item_ids,
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
          device = params.device,
          batch_size = params.batch_size,
          epochs = params.epochs, 
          lr = params.learning_rate, 
          print_freq = params.print_freq, 
          ckp_dir = params.ckp_dir)


    return result
