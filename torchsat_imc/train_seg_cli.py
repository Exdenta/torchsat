import imc_api_cli as imc_api
import enum
from train_seg import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, help='path to images', required=True)
    parser.add_argument('--labels_path', type=str, help='path to labels', required=True)
    parser.add_argument('--class_names', nargs='+', type=str, help='list of classes in the label dir', required=True)
    parser.add_argument('--mean', nargs='+', type=str, help='list of means for each channel in the dataset', required=True)
    parser.add_argument('--std', nargs='+', type=str, help='list of stds for each channel in the dataset', required=True)
    parser.add_argument('--use_gaussian_blur', type=bool, help='use gaussian blur', default=True)
    parser.add_argument('--gaussian_blur_kernel_size', default=3, type=int, help='size of gaussian blur kernel')
    parser.add_argument('--use_noise', type=bool, default=True)
    parser.add_argument('--noise_type', type=int, default=0)
    parser.add_argument('--noise_percent', default=0.02, type=float)
    parser.add_argument('--use_brightness', type=bool, default=True)
    parser.add_argument('--brightness_max_value', default=1, type=int, help='size of gaussian blur kernel')
    parser.add_argument('--use_contrast', type=bool, default=True)
    parser.add_argument('--contrast_max_value', default=1, type=int)
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
    parser.add_argument('--model_name', default="unet34", type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--resume_path', type=str, default="")
    parser.add_argument('--num_input_channels', default=3, type=int)
    parser.add_argument('--num_output_classes', default=3, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--dataset_split', default=0.1, type=float)
    parser.add_argument('--ckp_dir', default="./", type=str)

    args = parser.parse_args()

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
        brightness_max_value (int): max value of increasing and decreasing image brightness
        use_contrast (bool): use contrast dataset augmentation
        contrast_max_value (int): max value of increasing and decreasing image contrast
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
        model_name (str): model name
        pretrained (bool): load model pretrained weights
        resume_path (str): path to the latest checkpoint
        num_input_channels (int): input image channels
        num_output_classes (int): num of classes in the output
        device (str): 'cpu' of 'gpu', device to train model on
        batch_size (int): training batch size
        epochs (int): number of training epochs
        lr (float): initial training learning rate
        print_freq (int): metric values print frequency
        dataset_split (float): dataset train/validation split
        ckp_dir (str): path to save checkpoint
    """

    mean = [float(x) for x in args.mean]
    std = [float(x) for x in args.std]

    params = imc_api.TrainingParams(
        Path(args.features_path), 
        Path(args.labels_path),
        args.class_names,
        mean,
        std,
        args.use_gaussian_blur,
        args.gaussian_blur_kernel_size,
        args.use_noise,
        args.noise_type,
        args.noise_percent,
        args.use_brightness,
        args.brightness_max_value,
        args.use_contrast,
        args.contrast_max_value,
        args.use_shift,
        args.shift_max_percent,
        args.use_rotation,
        args.rotation_max_left_angle_value,
        args.rotation_max_right_angle_value,
        args.use_horizontal_flip,
        args.horizontal_flip_probability,
        args.use_vertical_flip,
        args.vertical_flip_probability,
        args.use_flip,
        args.flip_probability,
        args.crop_size,
        args.model_name,
        args.pretrained,
        args.resume_path,
        args.num_input_channels,
        args.num_output_classes,
        args.device,
        args.batch_size,
        args.epochs,
        args.lr,
        args.print_freq, 
        args.dataset_split,
        Path(args.ckp_dir),
    )

    train_segmentation(params=params, training_panel=None, progress_bar=None, called_from_imc=False)
