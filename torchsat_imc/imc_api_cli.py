"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-09-01 11:00:00
 * @modify date 2021-09-01 11:00:00
 * @desc mock declarations in case scripts are run not from IMC (without callbacks and embedded module)
"""

from enum import Enum
from pathlib import Path

class Device(Enum):
    """ Device type """
    CPU = 0,
    CUDA = 1,

    def __str__(self):
        return self.value

class NoiseType(Enum):
    """ Noise type """
    Gaussian = 0,
    Salt = 1,
    Pepper = 2

    def __str__(self):
        return self.value

class MessageTitle(Enum):
    """ Message title for logger """
    LogInfo = 0,
    LogError = 1,
    LogInitError = 2

    def __str__(self):
        return self.value

class UpdatePreviewParams():
    """ Preview update params for callback 
        Args:
            preview_path (Path): full path to a preview image
            preview_layer_name (str): new name for the document with preview image 
    """
    def __init__(self, preview_path: Path, preview_layer_name: str):
        self.preview_path = preview_path
        self.preview_layer_name = preview_layer_name

class InferenceParams():
    """ params for segmentation model inference 
    
        Args:
            image_path (Path): full path to image file
            model_path (Path): full path to pytorch state dict
            model_arch (str): model architecture name
            num_classes (int): number of output classes (1 class = 1 channel)
            channel_count (int): number of input channels (to take from each image for processing)
            tile_size (int): tile size
            device (imc_api.Device): hardware to run on
        """
    def __init__(self, image_path: Path, model_path: Path, model_arch: str, num_classes: int, channel_count: int, tile_size: int, device: Device):
        self.image_path = image_path
        self.model_path = model_path
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.channel_count = channel_count 
        self.tile_size = tile_size
        self.device = device

class TrainingParams():
    """ params for segmentation model training """
    def __init__(self, 
                 features_path: Path = Path(""), labels_path: Path = Path(""), label_classes: list = [],
                 preview_imagepath: Path = Path(""), preview_outdir: Path = Path(""),
                 mean: list = [], std: list = [],
                 use_gaussian_blur: bool = True, gaussian_blur_kernel_size: int = 3,
                 use_noise: bool = True, noise_type: NoiseType = NoiseType.Gaussian, noise_percent: float = 0.02,
                 use_brightness: bool = True, brightness_max_value: int = 1,
                 use_contrast: bool = True, contrast_max_value: int = 1,
                 use_shift: bool = True, shift_max_percent: float = 0.4,
                 use_rotation: bool = True, rotation_max_left_angle_value: int = -90, rotation_max_right_angle_value: int = 90,
                 use_horizontal_flip: bool = True, horizontal_flip_probability: float = 0.5,
                 use_vertical_flip: bool = True, vertical_flip_probability: float = 0.5,
                 use_flip: bool = True, flip_probability: float = 0.5,
                 crop_size: int = 128, model_name: str = "unet34",
                 pretrained: bool = False,
                 resume_path: Path = Path(""),
                 num_input_channels: int = 3,
                 num_output_classes: int = 3,
                 device: Device = Device.CPU,
                 batch_size: int = 16,
                 epochs: int = 90,
                 lr: float = 0.001,
                 print_freq: int = 10,
                 dataset_split: float = 0.8,
                 ckp_dir: Path = Path("./")):

        self.features_path = features_path
        self.labels_path = labels_path
        self.label_classes = label_classes
        self.mean = mean
        self.std = std
        self.preview_imagepath = preview_imagepath
        self.preview_outdir = preview_outdir
        self.use_gaussian_blur = use_gaussian_blur
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.use_noise = use_noise
        self.noise_type = noise_type
        self.noise_percent = noise_percent
        self.use_brightness = use_brightness
        self.brightness_max_value = brightness_max_value
        self.use_contrast = use_contrast
        self.contrast_max_value = contrast_max_value
        self.use_shift = use_shift
        self.shift_max_percent = shift_max_percent
        self.use_rotation = use_rotation
        self.rotation_max_left_angle_value = rotation_max_left_angle_value
        self.rotation_max_right_angle_value = rotation_max_right_angle_value
        self.use_horizontal_flip = use_horizontal_flip
        self.horizontal_flip_probability = horizontal_flip_probability
        self.use_vertical_flip = use_vertical_flip
        self.vertical_flip_probability = vertical_flip_probability
        self.use_flip = use_flip
        self.flip_probability = flip_probability
        self.crop_size = crop_size
        self.model_name = model_name
        self.pretrained = pretrained
        self.resume_path = resume_path
        self.num_input_channels = num_input_channels
        self.num_output_classes = num_output_classes
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.print_freq = print_freq
        self.dataset_split = dataset_split
        self.ckp_dir = ckp_dir

class ProgressBarPtr():
    """ pointer to progress bar for callbacks """
    def __init__(self):
        pass

class TrainingPanelPrt():
    """ pointer to training panel for callbacks """
    def __init__(self):
        pass

class SegmentationModelCheckpoint():
    """ Training segmentation model checkpoint """
    def __init__(self, 
                 checkpoint_name: str, 
                 date, 
                 training_lr: float, 
                 training_loss: float, 
                 validation_epoch_loss: float, 
                 validation_precision: float, 
                 validation_recall: float, 
                 validation_f1: float):
        
        self.checkpoint_name = checkpoint_name
        self.date = date
        self.training_lr = training_lr
        self.training_loss = training_loss
        self.validation_epoch_loss = validation_epoch_loss
        self.validation_precision = validation_precision
        self.validation_recall = validation_recall
        self.validation_f1 = validation_f1

def confirm_running(training_panel: TrainingPanelPrt):
    """ callback mock, confirm running """
    pass

def stop_training(training_panel: TrainingPanelPrt):
    """ callback mock, stop training """
    pass

def update_epoch(epoch: int, training_panel: TrainingPanelPrt):
    """ callback mock, update epoch """
    pass

def show_message(training_panel: TrainingPanelPrt, title: MessageTitle, message: str, message_to_log: str = ""):
    """ callback mock, show message to user """
    print(title.name, message, message_to_log)
    
def log_message(training_panel: TrainingPanelPrt, title: MessageTitle, message: str):
    """ callback mock, log message """
    print(title.name, message)

def update_progress(dProgressCounter: float, progress_bar: ProgressBarPtr):
    """ callback mock, update progress bar """
    pass

def update_preview_image(processed_preview_path: Path, training_panel: TrainingPanelPrt):
    """ callback mock, update preview image to show training progress """
    pass

def add_checkpoint(training_checkpoint: SegmentationModelCheckpoint, training_panel: TrainingPanelPrt):
    """ callback mock, add checkpoint to list """
    pass

def check_progress_bar_cancelled(progress_bar: ProgressBarPtr):
    """ callback mock, check progress bar status if its cancelled """
    pass
