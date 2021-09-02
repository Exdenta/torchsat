"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-08-30 11:00:00
 * @modify date 2021-08-30 11:00:00
 * @desc run segmentation model on image script
"""

import os
import json
import argparse
import datetime
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
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

try:
    import imc_api                     
except ImportError:
    import imc_api_cli as imc_api

from ignite.metrics import IoU, Precision, Recall # from pytorch-ignite

import torchsat_imc.transforms.transforms_seg as T_seg
from torchsat_imc.datasets.folder import SegmentationDataset
from torchsat_imc.models.utils import get_model
from torchsat_imc.scripts.make_mask_seg_onehot import split_images_and_labels
import torchsat_imc.imc_callbacks as imc_callbacks



def split_image_on_tiles(image_filepath: Path, label_classes: set, tile_size: int, drop_last: bool) -> np.array:
    """ Split image on tiles

        Args:
            image_filepath (Path): full path to image file
            label_classes (set): list of classes in the label
            tile_size (int): tile size
            drop_last (bool): drop last tiles in the edges of the image
    """

    class_count = len(label_classes)
    if class_count == 0:
        imc_callbacks.log_message(imc_api.MessageTitle.LogError, "No label classes were specified!")
        return False

    if not image_filepath.is_file():
        imc_callbacks.show_message(imc_api.MessageTitle.LogError, _("file {} does not exits.".format(image_filepath)))
        return False


    # split image and label
    img_src = rasterio.open(image_filepath)
    rows = img_src.meta['height'] // tile_size if drop_last else img_src.meta['height'] // tile_size + 1
    cols = img_src.meta['width']  // tile_size if drop_last else img_src.meta['width']  // tile_size + 1
    
    # allocate memory for tiles (tiles num, channel num, tile width, tile height)
    image_tiles = np.zeros((rows * cols, len(label_classes), tile_size, tile_size))

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            patched_arr = img_src.read(window=Window(col * tile_size, row * tile_size, tile_size, tile_size), boundless=True)
            image_tiles[idx] = patched_arr

    return image_tiles


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


def process_image(  training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr, 
                    current_progress: float, model: str = 'unet34', device: str = 'cpu') -> bool:
    """Process image with segmentation model
    
    Args:
        model (str): model name
        device (str): 'cpu' of 'gpu', device to run model on
    """

    result = True
    progress_step = current_progress / (epochs + 1)
    imc_callbacks.confirm_running(training_panel)
    imc_callbacks.update_progress(current_progress, _("Starting processing"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True
    current_progress += progress_step

    # try:

    torch.backends.cudnn.benchmark = False
    if device == 'cuda' and not torch.cuda.is_available():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("CUDA is not available. Falling back to CPU"))   
        device = 'cpu'
    
    device = torch.device('cuda' if device == 'cuda' else 'cpu')
    torch.cuda.empty_cache()

    if len(mean) != len(std):
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("The standart deviation array must be the same size as the mean array"))        
        return False

    if len(mean) != num_input_channels:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("Number of input channels must be the same as the size of mean and std arrays"))        
        return False

    # dataset and dataloader
    train_data, val_data = load_data(   features_dirpath, labels_dirpath, train_item_filenames, val_item_filenames, 
                                        mean, std,
                                        use_gaussian_blur, gaussian_blur_kernel_size,
                                        use_noise, noise_type, noise_percent,
                                        use_brightness, brightness_max_value,
                                        use_contrast, contrast_max_value,
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
    model = get_model(model, num_output_classes, pretrained=pretrained)
    model.to(device)

    # TODO: permission denied error if true
    if Path(resume_path).exists() and Path(resume_path).is_file():
        model.load_state_dict(torch.load(resume_path, map_location=device))

    # TODO: add losses choise
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
        training_metrics = train_one_epoch(epoch, train_loader, model, criterion, optimizer, device, writer, progress_step, current_progress, progress_bar, training_panel)
        if imc_callbacks.check_progress_bar_cancelled(progress_bar):
            return True
        validation_metrics = evaluation(epoch, val_loader, model, criterion, device, writer)
        lr_scheduler.step()

        # save checkpoint 
		# .def_readwrite("year", &DateTime::m_iYear)
		# .def_readwrite("month", &DateTime::m_iMon)
		# .def_readwrite("day", &DateTime::m_iDay)
		# .def_readwrite("hour", &DateTime::m_iHour)
		# .def_readwrite("min", &DateTime::m_iMin)
		# .def_readwrite("sec", &DateTime::m_iSec);

        checkpoint_name = f"cls_epoch_{epoch}"
        now = datetime.datetime.now()
        current_date = imc_api.DateTime(now.year, now.month, now.day, now.hour, now.minute, now.second)
        training_checkpoint = imc_api.SegmentationModelCheckpoint(checkpoint_name, current_date, learning_rate, training_metrics.loss, validation_metrics.loss, 
                                                            validation_metrics.precision, validation_metrics.recall, validation_metrics.f1)
        imc_callbacks.update_epoch(epoch, training_panel)
        imc_callbacks.add_checkpoint(training_checkpoint, training_panel)
        torch.save(model.state_dict(), os.path.join(ckp_dir, checkpoint_name + ".pth"))
    result = True

    return result

    # except Exception as e:
    #     imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("Training has failed"), str(e))
    #     result = False
    # finally:
    #     return result
            


def run_segmentation_alg(params: imc_api.InferenceParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr, called_from_imc: bool = True):
    """ Segmentation model inference on image, memory greedy but fast """

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
    
    splitted_dataset_path = params.features_path.parent / "dataset"
    rasterized_labels_path = splitted_dataset_path / "rasterized"
    features_outpath = splitted_dataset_path / "features"
    labels_outpath = splitted_dataset_path / "labels"

    train_config_path = splitted_dataset_path / "config.json"

    # create directories if don't exist

    for path in [splitted_dataset_path, rasterized_labels_path, features_outpath, labels_outpath]:
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
        previous_class_list = set([str(x) for x in config_json["classes"]])
        previous_dataset_items = config_json["items"] # dictionary ['item id': ('feature last update date', 'label last update date')]    

        # compare params of the last run with current

        # crop size
        if previous_crop_size != params.crop_size:
            crop_size_changed = True

        # classes
        new_classes_ids = set(params.label_classes) - set(previous_class_list)     # classes added since last run
        deleted_classes_ids = set(previous_class_list) - set(params.label_classes) # classes deleted since last run
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
                has_modified_items = True

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
                    has_modified_items = True
                    break

    modified_items_ids = modified_labels_ids.union(modified_features_ids)          # modified item ids (has modified label of feature)

    #
    # Split images and labels
    #

    imc_callbacks.update_progress(current_progress, _("Preparing the dataset"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True
    current_progress += progress_step

    if class_num_changed or crop_size_changed:
        # remove all features
        for child in features_outpath.iterdir():
            if child.is_file():
                child.unlink(True)
        # remove all labels
        for child in labels_outpath.iterdir():
            if child.is_file():
                child.unlink(True)
        # rasterize and split all labels and images
        split_images_and_labels(item_ids=dataset_item_ids, 
                                images_dirpath = params.features_path, 
                                labels_dirpath = params.labels_path, 
                                classes = params.label_classes, 
                                tile_size = params.crop_size, 
                                drop_last = drop_last, 
                                image_outdir = features_outpath, 
                                label_outdir = labels_outpath,
                                id_separator = id_separator,
                                tile_ext = tiles_extension) 

    elif item_num_changed or has_modified_items:
        # remove modified items
        for id_list in [deleted_item_ids, modified_items_ids]:
            delete_items(id_list, features_outpath, labels_outpath, id_separator)
        # rasterize and split new and modified labels and images
        split_images_and_labels(item_ids=new_item_ids.union(modified_items_ids), 
                                images_dirpath = params.features_path, 
                                labels_dirpath = params.labels_path, 
                                classes = params.label_classes, 
                                tile_size = params.crop_size, 
                                drop_last = drop_last, 
                                image_outdir = features_outpath, 
                                label_outdir = labels_outpath,
                                id_separator = id_separator,
                                tile_ext = tiles_extension) 
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

        intersecting_item_ids = splitted_features_ids_list.intersect(splitted_labels_ids_list)
        if len(intersecting_item_ids) <= 0:
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("No valid items in the dataset"))
            # imc_callbacks.stop_training(training_panel)
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
    config_json["classes"] = [str(x) for x in params.label_classes]
    config_json["items"] = dataset_items
    with open(train_config_path, 'w') as outfile:
        json.dump(config_json, outfile)

    # split dataset items on training and validation

    dataset_split_idx = int(len(splitted_dataset_item_ids) * params.dataset_split)
    
    train_dataset_item_ids = set(list(splitted_dataset_item_ids)[:dataset_split_idx])
    if len(train_dataset_item_ids) <= 0:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("No images in the dataset for training. Try adding some"))
        # imc_callbacks.stop_training(training_panel)
        return False

    val_dataset_item_ids = set(list(splitted_dataset_item_ids)[dataset_split_idx:])
    if len(val_dataset_item_ids) <= 0:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("No images in the dataset for validation. Try adding more or changing dataset split rate"))
        # imc_callbacks.stop_training(training_panel)
        return False

    imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogInfo, f"Using {len(train_dataset_item_ids)} images for training, {len(val_dataset_item_ids)} images for validation")

    # 
    # 4. Train on splitted dataset
    #
    
    result = train(
        training_panel = training_panel,
        progress_bar = progress_bar,
        current_progress = current_progress,
        features_dirpath = features_outpath, 
        labels_dirpath = labels_outpath,
        class_names = params.label_classes,
        train_item_filenames = train_dataset_item_ids,
        val_item_filenames = val_dataset_item_ids,
        mean = params.mean, 
        std = params.std,
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
        resume_path = params.resume_path, 
        num_input_channels = params.num_input_channels,
        num_output_classes = params.num_output_classes,
        device = params.device,
        batch_size = params.batch_size,
        epochs = params.epochs, 
        lr = params.lr, 
        print_freq = params.print_freq, 
        ckp_dir = params.ckp_dir)

    return result


def run_segmentation(params: imc_api.InferenceParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> bool:
    """ Segmentation model inference on image
    Args:
        params (imc_api.TrainingParams): training params
        training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks
        progress_bar (imc_api.ProgressBarPtr): progress bar ptr for progress updates
    """
    
    # progress update for progress bar
    current_progress = 0.0
    progress_step = 1.0
    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Setting up parameters for training"), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return True

    # process image
    try:
        return run_segmentation_alg()
    except MemoryError as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e) + ". Trying less memory-hungry algorithm...")
    
    return False
