"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-09-01 11:00:00
 * @modify date 2021-09-01 11:00:00
 * @desc mock declarations in case scripts are run not from IMC (without embedded module and callbacks)
"""

from enum import Enum, IntEnum
from pathlib import Path

class Device(Enum):
    """ Device type """
    CPU = 0,
    CUDA = 1,

    def __str__(self):
        return self.value

class ExtendedEnum(IntEnum):
    @classmethod
    def list(self):        
        role_names = [member.name for role, member in self.__members__.items()]
        return role_names

class LossFunction(ExtendedEnum):
    """ Loss function type """
    BCELoss = 0,
    DiceLoss = 1,
    DiceBCELoss = 2,
    IoULoss = 3,
    FocalLoss = 4, 
    TverskyLoss = 5,
    FocalTverskyLoss = 6

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
    LogWarning = 1,
    LogError = 2,
    LogInitError = 3
    LogProcessingError = 4,
    LogPreprocessingError = 5,
    LogPostprocessingError = 6,
    LogFilesystemError = 7,
    LogInvalidArgument = 8,
    LogUnknownError = 9

    def __str__(self):
        return self.value

class DateTime():
    """ DateTime utils class"""
    def __init__(self, year, month, day, hour, minute, second):
        self.year = year 
        self.month = month 
        self.day = day 
        self.hour = hour 
        self.minute = minute 
        self.second = second 

class UpdatePreviewParams():
    """ Preview update params for callback 
        Args:
            preview_path (Path): full path to a preview image
            preview_layer_name (str): new name for the document with preview image 
    """
    def __init__(self, preview_path: Path, preview_layer_name: str):
        self.preview_path = preview_path
        self.preview_layer_name = preview_layer_name

class SegmentationInferenceParams():
    """ params for segmentation model inference 
    
        Args:
            image_path (Path): full path to image file
            model_path (Path): full path to pytorch state dict
            model_arch (str): model architecture name
            num_classes (int): number of output classes (1 class = 1 channel)
            channel_count (int): model input size channel count
            tile_size (int): tile size
            device (imc_api.Device): hardware to run on
        """
    def __init__(self, image_path: Path, model_path: Path, preview_outdir: Path, model_arch: str,
                       num_classes: int, channel_count: int, tile_size: int, device: Device):
        self.image_path = image_path
        self.model_path = model_path
        self.preview_outdir = preview_outdir
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.channel_count = channel_count
        self.tile_size = tile_size
        self.device = device


class SegmentationTrainingParams():
    """ params for segmentation model training """
    def __init__(self, 
                 label_classes: list = [],
                 features_path: Path = Path(""), labels_path: Path = Path(""),
                 preview_imagepath: Path = Path(""), preview_outdir: Path = Path(""),
                 mean: list = [], std: list = [],
                 use_gaussian_blur: bool = True, gaussian_blur_kernel_size: int = 3,
                 use_noise: bool = True, noise_type: NoiseType = NoiseType.Gaussian, noise_percent: float = 0.02,
                 use_brightness: bool = True, brightness_max_percent: float = 0.1,
                 use_contrast: bool = True, contrast_max_percent: float = 0.1,
                 use_shift: bool = True, shift_max_percent: float = 0.4,
                 use_rotation: bool = True, rotation_max_left_angle_value: int = -90, rotation_max_right_angle_value: int = 90,
                 use_horizontal_flip: bool = True, horizontal_flip_probability: float = 0.5,
                 use_vertical_flip: bool = True, vertical_flip_probability: float = 0.5,
                 use_flip: bool = True, flip_probability: float = 0.5,
                 crop_size: int = 128, model_arch: str = "unet34",
                 pretrained: bool = False,
                 resume_path: Path = Path(""),
                 device: Device = Device.CPU,
                 loss_function: LossFunction = LossFunction.BCELoss,
                 batch_size: int = 16,
                 epochs: int = 90,
                 lr: float = 0.001,
                 print_freq: int = 10,
                 dataset_split: float = 0.8,
                 ckp_dir: Path = Path("./")):

        self.label_classes = label_classes
        self.features_path = features_path
        self.labels_path = labels_path
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
        self.brightness_max_percent = brightness_max_percent
        self.use_contrast = use_contrast
        self.contrast_max_percent = contrast_max_percent
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
        self.model_arch = model_arch
        self.pretrained = pretrained
        self.resume_path = resume_path
        self.device = device
        self.loss_function = loss_function
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
                 creation_date: DateTime, 
                 checkpoint_path: Path,
                 training_lr: float, 
                 training_loss: float, 
                 validation_epoch_loss: float, 
                 validation_precision: float, 
                 validation_recall: float, 
                 validation_f1: float):
        
        self.checkpoint_name = checkpoint_name
        self.creation_date = creation_date
        self.checkpoint_path = checkpoint_path
        self.training_lr = training_lr
        self.training_loss = training_loss
        self.validation_epoch_loss = validation_epoch_loss
        self.validation_precision = validation_precision
        self.validation_recall = validation_recall
        self.validation_f1 = validation_f1

class ConvertCheckpointParams:
    """ Params to convert pytorch checkpoint to onnx model"""
    def __init__(self, model_arch: str, model_path: Path, output_model_path: Path, input_channels: int, image_size: int, num_classes: int):
        self.model_arch = model_arch
        self.model_path = Path(model_path)
        self.output_model_path = output_model_path
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes

class OnnxModelParams:
    """ Params to add new onnx model after conversion from pytorch checkpoint 
    Args:
        model_path (Path): path to converted onnx model
        models_difference (float): max value difference between pytorch checkpoint and onnx model inference on test tensor
    """
    def __init__(self, onnx_model_path: Path, models_difference: float):
        self.model_path = onnx_model_path
        self.models_difference = models_difference


def confirm_running(training_panel: TrainingPanelPrt):
    """ callback mock, confirm running """
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

def add_model(params: OnnxModelParams, training_panel: TrainingPanelPrt):
    """ callback mock, add new onnx model to the list of segmentation models"""
    pass

def check_progress_bar_cancelled(progress_bar: ProgressBarPtr):
    """ callback mock, check progress bar status if its cancelled """
    pass
