import argparse
from torchsat_imc.scripts.train_seg import train_segmentation

def test():
    return "Hello world!"

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Training')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--val-path', help='validate dataset path')
    parser.add_argument('--mean', nargs='+', default=[0.485, 0.456, 0.406], type=float, help='dataset mean')
    parser.add_argument('--std', nargs='+', default=[0.229, 0.224, 0.225], type=float, help='dataset std')
    parser.add_argument('--image_extensions', nargs='+', default='jpg', help='image extension')
    parser.add_argument('--label_extension', default='png', help='label extension')
    parser.add_argument('--model', default="unet34", help='')
    parser.add_argument('--pretrained', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
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



def train2(train_path: str, val_path: str, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225], 
                       image_extensions: list = ('jpg'), label_extension: str = 'png', model: str = 'unet34', pretrained: bool = True,
                       resume: str = '', num_classes: int = 3, in_channels: int = 3, crop_size: int = 512, device: str = 'cpu', 
                       batch_size: int = 16, epochs: int = 90, lr: float = 0.001, print_freq: int = 10, ckp_dir: str = './'):

    print("Starting training segmentation model. Train path: ", train_path, ", Val path: ", val_path)
    return("Train 2 params: " + train_path + val_path)

    train_segmentation(
        train_path, 
        val_path, 
        mean, 
        std, 
        image_extensions, 
        label_extension, 
        model, 
        pretrained,
        resume, 
        num_classes, 
        in_channels, 
        crop_size, 
        device, 
        batch_size, 
        epochs, 
        lr, 
        print_freq, 
        ckp_dir
    )

def train():
    args = parse_args()

    train_segmentation(
        args.train_path, 
        args.val_path, 
        args.mean, 
        args.std, 
        args.image_extensions, 
        args.label_extension, 
        args.model, 
        args.pretrained,
        args.resume, 
        args.num_classes, 
        args.in_channels, 
        args.crop_size, 
        args.device, 
        args.batch_size, 
        args.epochs, 
        args.lr, 
        args.print_freq, 
        args.ckp_dir
    )

def main():
    train()

if __name__ == "__main__":
    main()
