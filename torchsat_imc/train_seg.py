"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 30-08-2021 11:00:00
 * @modify date 03-09-2021 11:00:00
 * @desc this tool is to split dataset images on tiles and train segmentation models on them
"""

# Notice when writing code! 
# Do not use set(A) - set(B) operation. It probably uses different minus operator
# when script starts from IMC, causing 2 sets to unite instead of subtract

import os
import gettext
_ = gettext.gettext
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
try:
    import imc_api                     
except ImportError:
    import imc_api_cli as imc_api
import torchsat_imc.imc_callbacks as imc_callbacks

import gc
import json
import argparse
import datetime
import rasterio
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import IoU, Precision, Recall # from pytorch-ignite
from torchsat_imc import run_seg
import torchsat_imc.transforms.transforms_seg as T_seg
from torchsat_imc.datasets.folder import SegmentationDataset
from torchsat_imc.imc_api_cli import TrainingPanelPrt
from torchsat_imc.models.utils import get_model
from torchsat_imc.utils import metrics
from torchsat_imc.utils import loss as loss_functions
from torchsat_imc.scripts.make_mask_seg_onehot import split_images_and_labels

shutdown_flag = False # True if user clicked 'Cancel' (for IMC integration only) 

def select_loss_function(loss_fn: imc_api.LossFunction) -> nn.Module:
    """ Return loss function pytorch module
    """
    if loss_fn == imc_api.LossFunction.BCELoss:
        return nn.BCELoss()
    elif loss_fn == imc_api.LossFunction.DiceBCELoss:
        return loss_functions.DiceBCELoss()
    elif loss_fn == imc_api.LossFunction.DiceLoss:
        return loss_functions.DiceLoss()
    elif loss_fn == imc_api.LossFunction.FocalLoss:
        return loss_functions.FocalLoss()
    elif loss_fn == imc_api.LossFunction.FocalTverskyLoss:
        return loss_functions.FocalTverskyLoss()
    elif loss_fn == imc_api.LossFunction.IoULoss:
        return loss_functions.IoULoss()
    elif loss_fn == imc_api.LossFunction.TverskyLoss:
        return loss_functions.TverskyLoss()
    else: 
        raise NotImplementedError("This loss function is not implemented!")


def delete_items(item_ids: set, splitted_image_dirpath: Path, splitted_labels_dirpath: Path, id_separator: str):
    """ Deletes item from the filesystem
    
        Args:
            item_ids (set): item ids to remove
            splitted_image_dirpath (Path): path to the directory with splitted images 
            splitted_labels_dirpath (Path): path to the directory with splitted reasterized labels
            id_separator (str): separator in the tiles filenames
    """

    # delete image tiles
    for image_filename in os.listdir(splitted_image_dirpath):
        image_parts = image_filename.split(id_separator)
        if len(image_parts) <= 0:
            continue
        image_id = image_parts[0]
        if image_id in item_ids:
            Path(splitted_image_dirpath / image_filename).unlink(True)

    # delete label tiles
    for label_filename in os.listdir(splitted_labels_dirpath):
        label_parts = label_filename.split(id_separator)
        if len(label_parts) <= 0:
            continue
        label_id = label_parts[0]
        if label_id in item_ids:
            Path(splitted_labels_dirpath / label_filename).unlink(True)


def train_one_epoch(epoch: int, dataloader: DataLoader, model, criterion: nn.Module, optimizer, device: str, writer: SummaryWriter, 
                    progress_step: float, current_progress: float, progress_bar: imc_api.ProgressBarPtr, training_panel: imc_api.TrainingPanelPrt) -> metrics.TrainingMetrics:
    """ Train 1 epoch

        Args:
            epoch (int): epoch number
            dataloader (DataLoader)
            criterion: loss function
            optimizer: optimization algorithm
            device (str): device to run on
            writer (SummaryWriter): tensorboard summary writer
            progress_step (float): progress bar step
            current_progress (float): current progress bar value
            progress_bar (imc_api.ProgressBarPtr): progress bar ptr for callbacks to IMC
            training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks to IMC
    """
    
    imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInfo, f"train epoch {epoch}")   
    progress_idx_step = progress_step / len(dataloader)

    model.train()
    softmax = nn.Softmax(dim=0)

    for idx, data in enumerate(dataloader):

        # if len(data.shape) == 3: # bug fix: in grayscale images

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
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInfo, 'train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx+1, len(dataloader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(dataloader)*epoch+idx)

        current_progress += progress_idx_step
        imc_callbacks.update_progress(current_progress, _("Training, epoch: ") + str(epoch), progress_bar)
        if imc_callbacks.check_progress_bar_cancelled(progress_bar):
            return None
    
    return metrics.TrainingMetrics(loss)


def process_preview(model, preview_imagepath: Path, channel_count: int, tile_size: int, device: str, output_dir: Path, output_filename: str, training_panel: TrainingPanelPrt) -> bool:
    """
    Process preview image and save to folder

        model: loaded Pytorch model to run
        preview_imagepath (pathlib.Path): path to the image for preview
        channel_count (int): input model channel count 
        tile_size (int): tile size to split image into (also model input size)
        device (str): device to run on
        output_dir (pathlib.Path): directory to save image to
        output_filename (str): preview image name
    """

    try:
        # process image
        image = run_seg.process_image(model=model, 
                                      image_path=preview_imagepath,
                                      channel_count=channel_count, 
                                      tile_size=tile_size, 
                                      device=device,
                                      drop_last=False,
                                      training_panel=training_panel,
                                      progress_bar=None)
        if image is None:
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogProcessingError, _("Failed to process the preview image. Model training will continue"))
            return False

        # save image
        filepath = output_dir / Path(output_filename + ".tif")
        image = (image * 255).astype(rasterio.uint8)
        with rasterio.open( filepath, 'w', driver='GTiff',
                            count=image.shape[0],
                            height=image.shape[1],
                            width=image.shape[2],
                            dtype=image.dtype,
                            # transform=rasterio.Affine.identity ### for some reason this doesn't work
                            ) as dst:
            dst.write(image)

        imc_callbacks.update_preview_image(imc_api.UpdatePreviewParams(filepath, output_filename), training_panel)

    except Exception as e:
        imc_api.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
        return False

    return True


def evaluation(epoch: int, dataloader: DataLoader, model, criterion: nn.Module, device: str, writer: SummaryWriter) -> metrics.ValidationMetrics:
    """
    Evaluation for onehot vector output segmentation

        epoch (int): epoch number
        dataloader (DataLoader)
        criterion: loss function
        optimizer: optimization algorithm
        device (str): device to run on
        writer (SummaryWriter): tensorboard summary writer
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

        # process validation data
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

    return metrics.ValidationMetrics(mean_loss_value, mean_precision, mean_recall, f1)


def load_data(  features_dirpath: Path, labels_dirpath: Path, train_item_filenames: set, val_item_filenames: set,
                mean: list, std: list,
                use_gaussian_blur: bool = True, gaussian_blur_kernel_size: int = 3,
                use_noise: bool = True, noise_type: imc_api.NoiseType = imc_api.NoiseType.Gaussian, noise_percent: float = 0.02,
                use_brightness: bool = True, brightness_max_percent: float = 0.1,
                use_contrast: bool = True, contrast_max_percent: float = 0.1,
                use_shift: bool = True, shift_max_percent: float = 0.4,
                use_rotation: bool = True, rotation_max_left_angle_value: int = -90, rotation_max_right_angle_value: int = 90,
                use_horizontal_flip: bool = True, horizontal_flip_probability: float = 0.5,
                use_vertical_flip: bool = True, vertical_flip_probability: float = 0.5,
                use_flip: bool = True, flip_probability: float = 0.5,
                crop_size: int = 128):
    """generate the train and val dataloader, you can change this for your specific task

    Args:
        features_dirpath (pathlib.Path): path to directory with features
        labels_dirpath (pathlib.Path): path to directory with labels
        train_item_filenames (set): filenames of training images to load from feature and label directories
        val_item_filenames (set): filenames of validation images to load from feature and label directories
        mean (list): dataset mean
        std (list): dataset std 
        use_gaussian_blur (bool): use gaussian blur dataset augmentation
        gaussian_blur_kernel_size (int): gaussian blue kernel size,
        use_noise (bool): use noise dataset augmentation
        noise_type (imc_api.NoiseType): type of the noise
        noise_percent (float): max noise percent
        use_brightness (bool): use brightness dataset augmentation
        brightness_max_percent (float): max percent of increasing and decreasing image brightness
        use_contrast (bool): use contrast dataset augmentation
        contrast_max_percent (float): max percent of increasing and decreasing image contrast
        use_shift (bool): use shift dataset augmentation
        shift_max_percent (float): from 0 to 1, shifts image randomly from 0 to this percent of image size
        use_rotation (bool): use rotation dataset augmentation
        rotation_max_left_angle_value (int): left rotation max angle (from -180 to 0)
        rotation_max_right_angle_value (int): right rotation max angle (from 0 to 180)
        use_horizontal_flip (bool): use horizontal flip dataset augmentation
        horizontal_flip_probability (float): probability of a flip
        use_vertical_flip (bool): use vertical flip dataset augmentation
        vertical_flip_probability (float): probability of a flip
        use_flip (bool): use flip dataset augmentation
        horizontal_flip_probability (float): probability of a flip
        crop_size (int): random crop size

    Returns:
        tuple: the train dataset and validation dataset
    """

    # dataset augmentation params for  

    train_transform = T_seg.Compose([
        T_seg.RandomCrop(crop_size),
    ])

    if use_gaussian_blur:
        train_transform.append(T_seg.GaussianBlur(kernel_size=gaussian_blur_kernel_size))

    if use_noise:
        noise_name = ""
        if noise_type == imc_api.NoiseType.Gaussian:
            noise_name = "gaussian"
        elif noise_type == imc_api.NoiseType.Pepper:
            noise_name = "pepper"
        elif noise_type == imc_api.NoiseType.Salt:
            noise_name = "salt"

        if noise_name != "":
            train_transform.append(T_seg.RandomNoise(mode=noise_name, percent=noise_percent))

    # TODO: convert brightness_max_percent and contrast_max_percent to max pixel value
    # and pass max pixel value to functions below:

    # if use_brightness:
    #     train_transform.append(T_seg.RandomBrightness(max_value=brightness_max_percent))

    # if use_contrast:
    #     train_transform.append(T_seg.RandomContrast(max_factor=contrast_max_percent))

    if use_shift:
        train_transform.append(T_seg.RandomShift(max_percent=shift_max_percent))

    if use_rotation:
        train_transform.append(T_seg.RandomRotation(degrees=[rotation_max_left_angle_value, rotation_max_right_angle_value]))

    if use_horizontal_flip:
        train_transform.append(T_seg.RandomHorizontalFlip(p=horizontal_flip_probability))

    if use_vertical_flip:
        train_transform.append(T_seg.RandomVerticalFlip(p=vertical_flip_probability))

    if use_flip:
        train_transform.append(T_seg.RandomFlip(p=flip_probability))

    train_transform.append(T_seg.ToTensor())
    train_transform.append(T_seg.Normalize(mean, std))


    # dataset augmentation params for validation  

    val_transform = T_seg.Compose([
        T_seg.ToTensor(),
        T_seg.Normalize(mean, std),
    ])

    dataset_train = SegmentationDataset(features_dirpath, labels_dirpath, train_item_filenames, transforms=train_transform)
    dataset_val = SegmentationDataset(features_dirpath, labels_dirpath, val_item_filenames, transforms=val_transform)

    return dataset_train, dataset_val


def _optimizer_to(optimizer, device):
    """ For deleting optimizer https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3"""
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def clear_memory(optimizer, model, device: str):
    # delete model from gpu memory
    if  device == 'cuda':
        _optimizer_to(optimizer, torch.device('cpu'))
    del optimizer
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def train(training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr, current_progress: float,
          features_dirpath: Path, labels_dirpath: Path, class_names: set,
          train_item_filenames: set, val_item_filenames: set,
          preview_imagepath: Path, preview_outdir : Path,
          mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225], 
          use_gaussian_blur: bool = True, gaussian_blur_kernel_size: int = 3,
          use_noise: bool = True, noise_type: int = 0, noise_percent: float = 0.02,
          use_brightness: bool = True, brightness_max_percent: float = 0.1,
          use_contrast: bool = True, contrast_max_percent: float = 0.1,
          use_shift: bool = True, shift_max_percent: float = 0.4,
          use_rotation: bool = True, rotation_max_left_angle_value: int = -90, rotation_max_right_angle_value: int = 90,
          use_horizontal_flip: bool = True, horizontal_flip_probability: float = 0.5,
          use_vertical_flip: bool = True, vertical_flip_probability: float = 0.5,
          use_flip: bool = True, flip_probability: float = 0.5,
          crop_size: int = 128, 
          model: str = 'unet34', pretrained: bool = True, 
          resume_path: Path = Path(""), device: imc_api.Device = imc_api.Device.CPU, loss_fn: imc_api.LossFunction = imc_api.LossFunction.BCELoss,
          batch_size: int = 16, epochs: int = 50, lr: float = 0.001, print_freq: int = 10, ckp_dir: Path = Path('./')) -> bool:
    """Training segmentation model
    
    Args:
        current_progress (float): current progress bar value
        features_dirpath (pathlib.Path): path to directory with features
        labels_dirpath (pathlib.Path): path to directory with labels
        class_names (set): list of label class names
        preview_image_path (Path): path to preview image to process after each epoch
        train_item_filenames (set): filenames of training images to load from feature and label directories
        val_item_filenames (set): filenames of validation images to load from feature and label directories
        preview_imagepath (pathlib.Path): path to the image for preview
        preview_outdir (pathlib.Path): path to the output directory for preview images
        mean (list): dataset mean
        std (list): dataset std 
        use_gaussian_blur (bool): use gaussian blur dataset augmentation
        gaussian_blur_kernel_size (int): gaussian blue kernel size,
        use_noise (bool): use noise dataset augmentation
        noise_type (imc_api.NoiseType): type of the noise
        noise_percent (float): max noise percent
        use_brightness (bool): use brightness dataset augmentation
        brightness_max_percent (float): max percent of increasing and decreasing image brightness
        use_contrast (bool): use contrast dataset augmentation
        contrast_max_percent (float): max percent of increasing and decreasing image contrast
        use_shift (bool): use shift dataset augmentation
        shift_max_percent (float): from 0 to 1, shifts image randomly from 0 to this percent of image size
        use_rotation (bool): use rotation dataset augmentation
        rotation_max_left_angle_value (int): left rotation max angle (from -180 to 0)
        rotation_max_right_angle_value (int): right rotation max angle (from 0 to 180)
        use_horizontal_flip (bool): use horizontal flip dataset augmentation
        horizontal_flip_probability (float): probability of a flip
        use_vertical_flip (bool): use vertical flip dataset augmentation
        vertical_flip_probability (float): probability of a flip
        use_flip (bool): use flip dataset augmentation
        horizontal_flip_probability (float): probability of a flip
        crop_size (int): random crop size
        model (str): model name
        pretrained (bool): load model pretrained weights
        resume_path (pathlib.Path): path to the latest checkpoint
        device (imc_api.Device): device type to train model on
        loss_fn (imc_api.LossFunction): loss function for training
        batch_size (int): training batch size
        epochs (int): number of training epochs
        lr (float): initial training learning rate
        print_freq (int): metric values print frequency
        ckp_dir (pathlib.Path): path to save checkpoint
    """

    result = True
    progress_step = current_progress / (epochs + 1)
    imc_callbacks.confirm_running(training_panel)
    imc_callbacks.update_progress(current_progress, _("Starting training"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True
    current_progress += progress_step

    torch.backends.cudnn.benchmark = False
    if (device == imc_api.Device.CUDA) and (not torch.cuda.is_available()):
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("CUDA is not available. Falling back to CPU"))   
        device = imc_api.Device.CPU
    
    device = torch.device('cuda' if device == imc_api.Device.CUDA else 'cpu')

    # check if can allocate memory on gpu
    if device == 'cuda':
        try:
            torch.cuda.empty_cache()
            t = torch.Tensor((1,2))
            t.to(device)
        except Exception as e:
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("CUDA is not available. Falling back to CPU"))  
            device = 'cpu'

    # try:

    if len(mean) != len(std):
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("The standart deviation array must be the same size as the mean array"))        
        return False

    # dataset and dataloader
    train_data, val_data = load_data(features_dirpath, labels_dirpath, train_item_filenames, val_item_filenames, 
                                     mean, std,
                                     use_gaussian_blur, gaussian_blur_kernel_size,
                                     use_noise, noise_type, noise_percent,
                                     use_brightness, brightness_max_percent,
                                     use_contrast, contrast_max_percent,
                                     use_shift, shift_max_percent,
                                     use_rotation, rotation_max_left_angle_value, rotation_max_right_angle_value,
                                     use_horizontal_flip, horizontal_flip_probability,
                                     use_vertical_flip, vertical_flip_probability,
                                     use_flip, flip_probability, crop_size)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # check cancelled
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True

    # model
    model = get_model(model, len(class_names), pretrained=pretrained)
    model.to(device)

    # TODO: permission denied error if true
    if Path(resume_path).exists() and Path(resume_path).is_file():
        model.load_state_dict(torch.load(resume_path, map_location=device))

    # loss function
    criterion = select_loss_function(loss_fn)

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    writer = SummaryWriter(ckp_dir)

    for epoch in range(epochs):
        learning_rate = lr_scheduler.get_lr()[0]
        writer.add_scalar('train/lr', learning_rate, epoch)
        
        # train
        training_metrics = train_one_epoch(epoch, train_loader, model, criterion, optimizer, device, writer, progress_step, current_progress, progress_bar, training_panel)
        if imc_callbacks.check_progress_bar_cancelled(progress_bar):
            clear_memory(optimizer, model, device)
            return True
        validation_metrics = evaluation(epoch, val_loader, model, criterion, device, writer)
        lr_scheduler.step()

        # save checkpoint
        now = datetime.datetime.now()
        checkpoint_name = f"cls_epoch_{epoch}_date_{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}"
        checkpoint_path  = ckp_dir / (checkpoint_name + ".pth")
        current_date = imc_api.DateTime(now.year, now.month, now.day, now.hour, now.minute, now.second)
        process_preview(model, preview_imagepath, len(mean), crop_size, device, preview_outdir, checkpoint_name, training_panel)
        training_checkpoint = imc_api.SegmentationModelCheckpoint(checkpoint_name, current_date, checkpoint_path, learning_rate, training_metrics.loss, validation_metrics.loss, 
                                                                  validation_metrics.precision, validation_metrics.recall, validation_metrics.f1)
        imc_callbacks.add_checkpoint(training_checkpoint, training_panel)
        torch.save(model.state_dict(), checkpoint_path)
    

    clear_memory(optimizer, model, device)
    result = True
    return result

    # except Exception as e:
    #     imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("Training has failed"), str(e))
    #     result = False
    # finally:
    #     return result


def continue_segmentation_training(params: imc_api.SegmentationTrainingParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> bool:
    """ Continue training segmentation model
    
    Args:
        params (imc_api.SegmentationTrainingParams): training params
        training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks
        progress_bar (imc_api.ProgressBarPtr): progress bar ptr for progress updates
    """


    # progress update for progress bar
    current_progress = 0.0
    progress_step = 1.0
    imc_callbacks.update_progress(current_progress, _("Setting up parameters for training"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True
    current_progress += progress_step

    splitted_dataset_path = params.features_path.parent / "dataset" # root dir for training data 
    features_outpath = splitted_dataset_path / "features"           # directory with all features splitted on tiles from prev run
    labels_outpath = splitted_dataset_path / "labels"               # directory with all labels splitted on tiles from prev run
    train_config_path = splitted_dataset_path / "config.json"       # config.json contains info about last run split configuration (to avoid unnecessary splitting)

    # check if directories don't exist
    for path in [splitted_dataset_path, features_outpath, labels_outpath]:
        if not path.exists():
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("Some directories with training data do not exist. Can't continue training!"))
            return False

    if not params.preview_outdir.exists():
        params.preview_outdir.mkdir()
    
    # get params from last training config

    if not train_config_path.exists():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("Can't continue training! No configuration from last training were saved"))
        return False

    # read params of the last run
    f = open(train_config_path)
    config_json = json.load(f)
    train_dataset_item_ids = config_json["training_items"] 
    val_dataset_item_ids = config_json["validation_items"]

    # start training
    result = train(
        training_panel = training_panel,
        progress_bar = progress_bar,
        current_progress = current_progress,
        features_dirpath = features_outpath, 
        labels_dirpath = labels_outpath,
        class_names = params.label_classes,
        train_item_filenames = train_dataset_item_ids,
        val_item_filenames = val_dataset_item_ids,
        preview_imagepath = params.preview_imagepath, 
        preview_outdir = params.preview_outdir,
        mean = params.mean, 
        std = params.std,
        use_gaussian_blur = params.use_gaussian_blur, gaussian_blur_kernel_size = params.gaussian_blur_kernel_size,
        use_noise = params.use_noise, noise_type = params.noise_type, noise_percent = params.noise_percent,
        use_brightness = params.use_brightness, brightness_max_percent = params.brightness_max_percent,
        use_contrast = params.use_contrast, contrast_max_percent = params.contrast_max_percent,
        use_shift = params.use_shift, shift_max_percent = params.shift_max_percent,
        use_rotation = params.use_rotation, rotation_max_left_angle_value = params.rotation_max_left_angle_value, rotation_max_right_angle_value = params.rotation_max_right_angle_value,
        use_horizontal_flip = params.use_horizontal_flip, horizontal_flip_probability = params.horizontal_flip_probability, 
        use_vertical_flip = params.use_vertical_flip, vertical_flip_probability = params.vertical_flip_probability,
        use_flip = params.use_flip, flip_probability = params.flip_probability, 
        crop_size = params.crop_size,
        model = params.model_arch,
        pretrained = params.pretrained,
        resume_path = params.resume_path, 
        device = params.device,
        batch_size = params.batch_size,
        epochs = params.epochs, 
        lr = params.lr, 
        print_freq = params.print_freq, 
        ckp_dir = params.ckp_dir)

    return result


def start_segmentation_training(params: imc_api.SegmentationTrainingParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> bool:
    """Training segmentation model
    
    Args:
        params (imc_api.SegmentationTrainingParams): training params
        training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks
        progress_bar (imc_api.ProgressBarPtr): progress bar ptr for progress updates
    """
    
    # progress update for progress bar
    current_progress = 0.0
    progress_step = 1.0
    imc_callbacks.update_progress(current_progress, _("Setting up parameters for training"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True
    current_progress += progress_step

    # try:

    # extension of tiles
    tiles_extension = ".tif"
    id_separator = "__"
    drop_last = False

    # changes to the dataset since last run
    item_num_changed = False      # number of items changed
    class_num_changed = False     # number of classes changed
    crop_size_changed = False     # crop size parameter changed
    has_modified_items = False    # some items were modified

    modified_features_ids = set() # item ids with modified features
    modified_labels_ids = set()   # item ids with modified labels    
    modified_items_ids = set()    # item ids with modified label or feature
    new_item_ids = set()          # added items
    new_classes_ids = set()       # added classes
    deleted_item_ids = set()      # deleted items
    deleted_classes_ids = set()   # deleted classes
    
    splitted_dataset_path = params.features_path.parent / "dataset" # root dir for training data 
    features_outpath = splitted_dataset_path / "features"           # directory to save all features splitted on tiles
    labels_outpath = splitted_dataset_path / "labels"               # directory to save all labels splitted on tiles
    train_config_path = splitted_dataset_path / "config.json"       # config.json contains info about last run split configuration (to avoid unnecessary splitting)
    temp_config_path = splitted_dataset_path / "config_copy.json"   # copy of config.json to be able to restore it

    label_classes = set([str(x) for x in params.label_classes])

    # create directories if don't exist

    for path in [splitted_dataset_path, features_outpath, labels_outpath, params.preview_outdir]:
        if not path.exists():
            path.mkdir()

    dataset_item_ids = set(os.listdir(params.labels_path)) # notice: dir names don't have extensions 

    # get params from last training config

    first_run = False # first program run for the dataset

    if not train_config_path.exists():
        train_config_path.touch()
        first_run = True
    else:
        try:
            # read params of the last run
            with open(train_config_path) as f:
                config_json = json.load(f)
                previous_crop_size = config_json["crop_size"]
                previous_class_list = set([str(x) for x in config_json["classes"]])
                previous_dataset_items = config_json["items"] # dictionary ['item id': ('feature last update date', 'label last update date')]    
                first_run = False
        except Exception as e:
            imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInitError, "Failed to read dataset configuration file. Exception occured: " + str(e))
            first_run = True

        # if read dataset config from last run successfully
        # compare params of the last run with current
        if not first_run:

            # crop size
            if previous_crop_size != params.crop_size:
                crop_size_changed = True

            # classes
            previous_class_list = set([str(x) for x in previous_class_list])
            new_classes_ids = label_classes.difference(previous_class_list)     # classes added since last run
            deleted_classes_ids = previous_class_list.difference(label_classes) # classes deleted since last run
            if len(deleted_classes_ids) != 0 or len(new_classes_ids) != 0:
                class_num_changed = True

            # items
            new_item_ids = dataset_item_ids - set(previous_dataset_items.keys())      # item ids added since last run
            deleted_item_ids = set(previous_dataset_items.keys()) - dataset_item_ids  # item ids deleted since last run
            if len(deleted_item_ids) != 0 or len(new_item_ids) != 0:
                item_num_changed = True

            # get items with modified features (images)
            for feature_filename in os.listdir(params.features_path):
                item_id = Path(feature_filename).stem
                previous_update_date = previous_dataset_items[item_id]
                if previous_update_date == None:
                    continue
                # if prev run update date is earlier - add to modified list 
                stat = os.stat(params.features_path / feature_filename)
                if stat.st_mtime > previous_update_date[0]:
                    modified_features_ids.add(item_id)

            # get items with modified labels (.geojson data)
            common_item_ids = dataset_item_ids.intersection(set(previous_dataset_items.keys()))
            for item_id in common_item_ids:
                previous_update_date = previous_dataset_items.get(item_id)
                if previous_update_date == None:
                    continue
                label_dirpath = params.labels_path / item_id
                for label in os.listdir(label_dirpath):
                    stat = os.stat(label_dirpath / label)
                    # if at least one class modified - add to modified list 
                    if stat.st_mtime > previous_update_date[1]:
                        modified_labels_ids.add(item_id)
                        break

    # modified item ids = (items with features modified since last run) + (items with labels modified since last run)
    modified_items_ids = modified_labels_ids.union(modified_features_ids) 
    has_modified_items = (len(modified_items_ids) != 0)
    
    #
    # Split images and labels
    #

    imc_callbacks.update_progress(current_progress, _("Preparing the dataset"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True
    current_progress += progress_step

    if first_run or crop_size_changed or class_num_changed:
        # remove all labels
        for child in labels_outpath.iterdir():
            if child.is_file():
                child.unlink(True)

        # if only number of classes changed, no need to delete images
        if class_num_changed and not (first_run or crop_size_changed):
            # rasterize and split all labels
            split_images_and_labels(item_ids=dataset_item_ids, 
                                    images_dirpath = params.features_path, 
                                    labels_dirpath = params.labels_path, 
                                    classes = label_classes, 
                                    tile_size = params.crop_size, 
                                    drop_last = drop_last, 
                                    image_outdir = features_outpath, 
                                    label_outdir = labels_outpath,
                                    id_separator = id_separator,
                                    tile_ext = tiles_extension,
                                    split_images=False,
                                    split_labels=True) 
        else:
            # remove all features
            for child in features_outpath.iterdir():
                if child.is_file():
                    child.unlink(True)
            # rasterize and split all labels and images
            split_images_and_labels(item_ids=dataset_item_ids, 
                                    images_dirpath = params.features_path, 
                                    labels_dirpath = params.labels_path, 
                                    classes = label_classes, 
                                    tile_size = params.crop_size, 
                                    drop_last = drop_last, 
                                    image_outdir = features_outpath, 
                                    label_outdir = labels_outpath,
                                    id_separator = id_separator,
                                    tile_ext = tiles_extension,
                                    split_images=True,
                                    split_labels=True) 
    elif item_num_changed or has_modified_items:
        # remove modified and deleted items
        for id_list in [deleted_item_ids, modified_items_ids]:
            delete_items(id_list, features_outpath, labels_outpath, id_separator)
        # rasterize and split new and modified labels and images
        split_images_and_labels(item_ids=new_item_ids.union(modified_items_ids), 
                                images_dirpath = params.features_path, 
                                labels_dirpath = params.labels_path, 
                                classes = label_classes, 
                                tile_size = params.crop_size, 
                                drop_last = drop_last, 
                                image_outdir = features_outpath, 
                                label_outdir = labels_outpath,
                                id_separator = id_separator,
                                tile_ext = tiles_extension,
                                split_images=True,
                                split_labels=True)  
    else:
        pass

    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True

    #
    # Delete all tiles with more than 50% of 'black' area 
    #

    for imagename in os.listdir(features_outpath):
        try:
            # image and label tile names are the same
            image_path = Path(features_outpath / imagename)
            label_path = Path(labels_outpath / imagename) 
            # read image
            image = rasterio.open(image_path).read()
            zero_pixel_count = np.count_nonzero(image==0)
            total_pixel_count = 1
            for i in range(len(image.shape)):
                total_pixel_count *= image.shape[i]
            # delete
            if zero_pixel_count > total_pixel_count // 2:
                image_path.unlink(True)
                label_path.unlink(True)

        except Exception as e:
            imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
            continue

    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True

    # check that number of feature tiles is the same as number of label tiles

    ext_len = len(tiles_extension)
    splitted_dataset_item_ids = []
    splitted_features_ids_list = set([filename[:-ext_len] for filename in os.listdir(features_outpath)])
    splitted_labels_ids_list = set([filename[:-ext_len] for filename in os.listdir(labels_outpath)])
    if len(splitted_features_ids_list) != len(splitted_labels_ids_list):
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInfo, f"Number of feature tiles ({len(splitted_features_ids_list)}) is not the same as number of label tiles ({len(splitted_labels_ids_list)})! Trying to fix...")

        # try to take only features and labels with intersecting names

        intersecting_item_ids = splitted_features_ids_list.intersection(splitted_labels_ids_list)
        if len(intersecting_item_ids) <= 0:
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("No valid items in the dataset"))
            return False
        
        splitted_dataset_item_ids = intersecting_item_ids

        # delete all non-intersecting names

        unique_feature_ids = (splitted_features_ids_list - splitted_labels_ids_list)
        for id in unique_feature_ids:
            (features_outpath / id).with_suffix(tiles_extension).unlink(True)

        unique_label_ids = (splitted_labels_ids_list - splitted_features_ids_list)
        for id in unique_label_ids:
            (labels_outpath / id).with_suffix(tiles_extension).unlink(True)
    else:
        splitted_dataset_item_ids = splitted_labels_ids_list

    # update train config
    
    dataset_items = dict()

    for id in dataset_item_ids:
        
        feature_filename = ""
        feature_found = False
        for filename in os.listdir(params.features_path):
            if Path(filename).stem == id:
                feature_filename = filename
                feature_found = True
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

    # split dataset items on training and validation

    dataset_split_idx = int(len(splitted_dataset_item_ids) * params.dataset_split)
    
    train_dataset_item_ids = set(list(splitted_dataset_item_ids)[:dataset_split_idx])
    if len(train_dataset_item_ids) <= 0:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("No images in the dataset for training. Try adding some"))
        return False

    val_dataset_item_ids = set(list(splitted_dataset_item_ids)[dataset_split_idx:])
    if len(val_dataset_item_ids) <= 0:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("No images in the dataset for validation. Try adding more or changing dataset split rate"))
        return False

    imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInfo, f"Using {len(train_dataset_item_ids)} images for training, {len(val_dataset_item_ids)} images for validation")

    # 
    # Update training config
    #

    # rename old config to be able to restore it
    temp_config_path.unlink(True)
    train_config_path.rename(temp_config_path)
    try:
        config_json = {}
        config_json["crop_size"] = params.crop_size
        config_json["classes"] = [str(x) for x in label_classes]
        config_json["items"] = dataset_items # TODO: can be optimized out
        config_json["training_items"] = list(train_dataset_item_ids) # to be able to continue training
        config_json["validation_items"] = list(val_dataset_item_ids) # to be able to continue training 
        with open(train_config_path, 'w') as outfile:
            json.dump(config_json, outfile)
    except Exception as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInfo, "Unable to save training configuration file to the disk! Dataset preparation operations will repeat during next training run.")
        train_config_path.unlink(True)
        temp_config_path.rename(train_config_path) # restore previous config

    # 
    # 4. Train on splitted dataset
    #

    result = train(
        training_panel = training_panel,
        progress_bar = progress_bar,
        current_progress = current_progress,
        features_dirpath = features_outpath, 
        labels_dirpath = labels_outpath,
        class_names = label_classes,
        train_item_filenames = train_dataset_item_ids,
        val_item_filenames = val_dataset_item_ids,
        preview_imagepath = params.preview_imagepath, 
        preview_outdir = params.preview_outdir,
        mean = params.mean, 
        std = params.std,
        use_gaussian_blur = params.use_gaussian_blur, gaussian_blur_kernel_size = params.gaussian_blur_kernel_size,
        use_noise = params.use_noise, noise_type = params.noise_type, noise_percent = params.noise_percent,
        use_brightness = params.use_brightness, brightness_max_percent = params.brightness_max_percent,
        use_contrast = params.use_contrast, contrast_max_percent = params.contrast_max_percent,
        use_shift = params.use_shift, shift_max_percent = params.shift_max_percent,
        use_rotation = params.use_rotation, rotation_max_left_angle_value = params.rotation_max_left_angle_value, rotation_max_right_angle_value = params.rotation_max_right_angle_value,
        use_horizontal_flip = params.use_horizontal_flip, horizontal_flip_probability = params.horizontal_flip_probability, 
        use_vertical_flip = params.use_vertical_flip, vertical_flip_probability = params.vertical_flip_probability,
        use_flip = params.use_flip, flip_probability = params.flip_probability, 
        crop_size = params.crop_size,
        model = params.model_arch,
        pretrained = params.pretrained,
        resume_path = params.resume_path, 
        device = params.device,
        loss_fn = params.loss_function,
        batch_size = params.batch_size,
        epochs = params.epochs, 
        lr = params.lr, 
        print_freq = params.print_freq, 
        ckp_dir = params.ckp_dir)
    
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    """Training segmentation model
    
    Args:
        features_dirpath (pathlib.Path): path to directory with features
        labels_dirpath (pathlib.Path): path to directory with labels
        class_names (set): list of label class names
        train_item_filenames (set): filenames of training images to load from feature and label directories
        val_item_filenames (set): filenames of validation images to load from feature and label directories
        mean (list): dataset mean
        std (list): dataset std 
        use_gaussian_blur (bool): use gaussian blur dataset augmentation
        gaussian_blur_kernel_size (int): gaussian blue kernel size,
        use_noise (bool): use noise dataset augmentation
        noise_type (imc_api.NoiseType): type of the noise
        noise_percent (float): max noise percent
        use_brightness (bool): use brightness dataset augmentation
        brightness_max_percent (float): max value of increasing and decreasing image brightness
        use_contrast (bool): use contrast dataset augmentation
        contrast_max_percent (float): max value of increasing and decreasing image contrast
        use_shift (bool): use shift dataset augmentation
        shift_max_percent (float): from 0 to 1, shifts image randomly from 0 to this percent of image size
        use_rotation (bool): use rotation dataset augmentation
        rotation_max_left_angle_value (int): left rotation max angle (from -180 to 0)
        rotation_max_right_angle_value (int): right rotation max angle (from 0 to 180)
        use_horizontal_flip (bool): use horizontal flip dataset augmentation
        horizontal_flip_probability (float): probability of a flip
        use_vertical_flip (bool): use vertical flip dataset augmentation
        vertical_flip_probability (float): probability of a flip
        use_flip (bool): use flip dataset augmentation
        flip_probability (float): probability of a flip
        crop_size (int): random crop size
        model_arch (str): model architecture
        pretrained (bool): load model pretrained weights
        resume_path (str): path to the latest checkpoint
        num_input_channels (int): input image channels
        device (str): 'cpu' of 'gpu', device to train model on
        loss_function (str): loss function from 'imc_api_cli.LossFunction' enumeration
        batch_size (int): training batch size
        epochs (int): number of training epochs
        lr (float): initial training learning rate
        print_freq (int): metric values print frequency
        dataset_split (float): dataset train/validation split
        ckp_dir (str): path to save checkpoint
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, help='path to images', required=True)
    parser.add_argument('--labels_path', type=str, help='path to labels', required=True)
    parser.add_argument('--class_names', nargs='+', type=str, help='list of classes in the label dir', required=True)
    parser.add_argument('--preview_imagepath', type=str, default="", help='path to the preview image to show after each epoch')
    parser.add_argument('--preview_outdir', type=str, default="", help='path to the preview images output folder')
    parser.add_argument('--mean', nargs='+', type=str, help='list of means for each channel in the dataset', required=True)
    parser.add_argument('--std', nargs='+', type=str, help='list of stds for each channel in the dataset', required=True)
    parser.add_argument('--use_gaussian_blur', type=bool, help='use gaussian blur', default=True)
    parser.add_argument('--gaussian_blur_kernel_size', default=3, type=int, help='size of gaussian blur kernel')
    parser.add_argument('--use_noise', type=bool, default=True)
    parser.add_argument('--noise_type', type=int, default=0)
    parser.add_argument('--noise_percent', default=0.02, type=float)
    parser.add_argument('--use_brightness', type=bool, default=True)
    parser.add_argument('--brightness_max_percent', default=0.1, type=float, help='size of gaussian blur kernel')
    parser.add_argument('--use_contrast', type=bool, default=True)
    parser.add_argument('--contrast_max_percent', default=0.1, type=float)
    parser.add_argument('--use_shift', default=True, type=bool)
    parser.add_argument('--shift_max_percent', default=0.4, type=float)
    parser.add_argument('--use_rotation', type=bool, default=True)
    parser.add_argument('--rotation_max_left_angle_value', type=int, default=-90)
    parser.add_argument('--rotation_max_right_angle_value', type=int, default=90)
    parser.add_argument('--use_horizontal_flip', default=True, type=bool)
    parser.add_argument('--horizontal_flip_probability', type=float, default=0.5,)
    parser.add_argument('--use_vertical_flip', default=True, type=bool)
    parser.add_argument('--vertical_flip_probability', type=float, default=0.5)
    parser.add_argument('--use_flip', default=True, type=bool)
    parser.add_argument('--flip_probability', type=float, default=0.5)
    parser.add_argument('--crop_size', default=128, type=int)
    parser.add_argument('--model_arch', default="unet34", type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--resume_path', type=str, default="")
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--loss_function', default="BCELoss", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--dataset_split', default=0.1, type=float)
    parser.add_argument('--ckp_dir', default="./", type=str)
    args = parser.parse_args()

    # classes
    args.class_names = [Path(x) for x in args.class_names]

    # get device
    device = imc_api.Device.CPU if args.device == 'cpu' else imc_api.Device.CUDA
    
    # get loss function
    loss_function = imc_api.LossFunction.BCELoss
    for idx, loss_name in enumerate(imc_api.LossFunction.list()):
        if loss_name == args.loss_function:
            loss_function = imc_api.LossFunction(idx)
            break

    # form training params
    params = imc_api.SegmentationTrainingParams(
        args.class_names,
        Path(args.features_path), Path(args.labels_path),
        Path(args.preview_imagepath), Path(args.preview_outdir),
        [float(x) for x in args.mean], [float(x) for x in args.std],
        args.use_gaussian_blur, args.gaussian_blur_kernel_size,
        args.use_noise, args.noise_type, args.noise_percent,
        args.use_brightness, args.brightness_max_percent,
        args.use_contrast, args.contrast_max_percent,
        args.use_shift, args.shift_max_percent,
        args.use_rotation, args.rotation_max_left_angle_value, args.rotation_max_right_angle_value,
        args.use_horizontal_flip, args.horizontal_flip_probability,
        args.use_vertical_flip, args.vertical_flip_probability,
        args.use_flip, args.flip_probability,
        args.crop_size,
        args.model_arch,
        args.pretrained,
        Path(args.resume_path),
        device, loss_function,
        args.batch_size,
        args.epochs,
        args.lr,
        args.print_freq, 
        args.dataset_split,
        Path(args.ckp_dir)
    )

    start_segmentation_training(params=params, training_panel=None, progress_bar=None)

