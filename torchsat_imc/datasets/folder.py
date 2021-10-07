# original source code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
import os
import os.path
import sys
from pathlib import Path
import numpy as np
import torch

import torch.utils.data as data

from .utils import default_loader, image_loader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        classes (callable, optional): List of the class names.
        class_to_idx (callable, optional): Dict with items (class_name, class_index).
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, classes=None, class_to_idx=None, transform=None, target_transform=None):
        if not class_to_idx:
            classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, **kwargs):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform, **kwargs)
        self.imgs = self.samples


class ChangeDetectionDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
            .
        ├── train
        │   ├── pre
        │   │   ├── train_1.png
        │   │   ├── train_2.png
        │   │   ├── ...
        │   ├── post
        │   │   ├── train_1.png
        │   │   ├── train_2.png
        │   │   ├── ...
        │   └── label
        │       ├── train_1.png
        │       ├── train_2.png
        │       ├── ...
        └── val
            ├── pre
            │   ├── val_10.png
            │   ├── val_11.png
            │   ├── ...
            ├── post
            │   ├── val_10.png
            │   ├── val_11.png
            │   ├── ...
            └── label
                ├── val_10.png
                ├── val_11.png
                ├── ...

    Args:
        root (string): root dir of train or validate dataset.
        extensions (tuple or list): extension of training image.
    """
    def __init__(self, root, image_extensions=('jpg'), label_extension='png', transform=None):
        self.root = root
        self.image_extensions = image_extensions
        self.label_extension = label_extension
        self.transform = transform

        self.samples = self._generate_data()

    def __getitem__(self, index):
        pre_img, post_img, label_img = [image_loader(x) for x in self.samples[index]]
        if self.transform is not None:
            pre_img, post_img, label_img = self.transform(pre_img, post_img, label_img)
        return pre_img, post_img, label_img

    def _generate_data(self):
        images = []
        for root, _, fnames in sorted(os.walk(os.path.join(self.root, 'pre'))):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, self.image_extensions):
                    pre_path = os.path.join(root, fname)
                    post_path = pre_path.replace('pre', 'post')
                    label_path = str(Path(pre_path.replace('pre', 'label')).with_suffix(self.label_extension))
                    images.append((pre_path, post_path, label_path))

        return images

    def __len__(self):
        return len(self.samples)


class SegmentationDataset(object):
    """A generic data loader where the images are arranged in this way: ::
        .
        ├── features
        │   ├── train_1.png
        │   ├── train_2.png
        │   ├── ...
        │   ├── valid_1.png
        │   ├── valid_2.png
        │   └── ...
        │  
        └── labels
            ├── val_10.png
            ├── val_11.png
            ├── ...
            ├── val_10.png
            ├── val_11.png
            └── ...

    Args:
        features_dirpath (pathlib.Path): path to directory with features
        labels_dirpath (pathlib.Path): path to directory with labels
        item_filenames (set): filenames of images to load from feature and label directories
        transforms: transformation for images in the dataset
    """

    def __init__(self, features_dirpath: Path, labels_dirpath: Path, item_filenames: set, transforms=None):
        self.features_dirpath = features_dirpath
        self.labels_dirpath = labels_dirpath
        self.item_filenames = item_filenames
        self.transforms = transforms

        self.samples = self._generate_data()
        pass

    def __getitem__(self, index):
        image_img, label_img = [image_loader(x) for x in self.samples[index]]
        if self.transforms is not None:
            image_img, label_img = self.transforms(image_img, label_img)

        # note: if label is a grayscale image with 1 channel (1 class in the dataset)
        # extra channel is ommited (shape is [128, 128] instead of [128, 128, 1])
        # this makes algorithm fail on this type of images during training
        # so: adding extra channel
        if len(label_img.shape) == 2:
            label_img = torch.unsqueeze(label_img, dim=2)

        return image_img, label_img

    def _generate_data(self):
        images = []
        files_in_directory = os.listdir(self.features_dirpath)
        for filename in files_in_directory:
            if Path(filename).stem in self.item_filenames: 
                image_path = self.features_dirpath / filename
                label_path = self.labels_dirpath / filename
                images.append((image_path, label_path))
        return images

    def __repr__(self):
        return "__init__(self, features_dirpath, labels_dirpath, item_filenames, transforms=None)"

    def __len__(self):
        return len(self.samples)


class SegmentationDatasetDir(object):
    """A generic data loader where the images are arranged in this way: ::
        .
        ├── train
        │   ├── image
        │   │   ├── train_1.png
        │   │   ├── train_2.png
        │   │   ├── ...
        │   └── label
        │       ├── train_1.png
        │       ├── train_2.png
        │       ├── ...
        └── val
            ├── image
            │   ├── val_10.png
            │   ├── val_11.png
            │   ├── ...
            └── label
                ├── val_10.png
                ├── val_11.png
                ├── ...

    Args:
        root (string): root dir of train or validate dataset.
        image_extensions (tuple or list): extension of image.
        label_extension (str): extension of label
    """

    def __init__(self, root, image_extensions=('jpg'), label_extension='png', transforms=None):
        self.root = root
        self.image_extensions = image_extensions
        self.label_extension = label_extension
        self.transforms = transforms

        self.samples = self._generate_data()
        pass

    def __getitem__(self, index):
        image_img, label_img = [image_loader(x) for x in self.samples[index]]
        if self.transforms is not None:
            image_img, label_img = self.transforms(image_img, label_img)
        return image_img, label_img

    def _generate_data(self):
        images = []
        for root, _, fnames in sorted(os.walk(os.path.join(self.root, 'image'))):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, self.image_extensions):
                    image_path = os.path.join(root, fname)
                    suffix = ("." if self.label_extension[0] != "." else "") + self.label_extension
                    label_path = Path(image_path.replace('image', 'label')).with_suffix(suffix)
                    images.append((image_path, label_path))

        return images

    def __repr__(self):
        return "__init__(self, root, image_extensions=('jpg'), label_extension='png', transforms=None)"

    def __len__(self):
        return len(self.samples)
