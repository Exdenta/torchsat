
import cv2
import numpy as np
import pathlib
from PIL import Image
from itertools import product
from pathlib import Path
from torchsat_imc.scripts.make_mask_seg_onehot import make_mask_seg

def rasterize_image(image_path: pathlib.Path):
    """ Image rasterization
    """
    pass

def split_image_on_tiles(image, image_name: str, tile_size: int, output_extension: str):
    """ Split image on tiles
    
        Args:
            image (np.array) image matrix
            image_name (str) image name for saving tiles
            tile_size (int) tile size
            output_extension (str): output tiles extension
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    for r in range(0, image_height, tile_size):
        for c in range(0, image_width, tile_size):
            cv2.imwrite(f"{image_name}_{r}_{c}.{output_extension}", image[r:r+tile_size, c:c+tile_size,:])


def split_dataset(features_path: Path, labels_path: Path, dataset_split: float, tile_size: int, tiles_extension: str) -> tuple(Path, Path):
    """ Splits dataset into training and validation directories for SegmentationDataset
            - rasterizes labels
            - splits images on tiles

        Folder structure after splitting:

        ├── 'train_path'
        │   ├── image
        │   │   ├── train_1.png
        │   │   ├── train_2.png
        │   │   ├── ...
        │   └── label
        │       ├── train_1.png
        │       ├── train_2.png
        │       ├── ...
        └── 'val_path'
            ├── image
            │   ├── val_10.png
            │   ├── val_11.png
            │   ├── ...
            └── label
                ├── val_10.png
                ├── val_11.png
                ├── ...

        Args:
            features_path (pathlib.Path): path to directory with feature images
            labels_path (pathlib.Path): path to directory with other labels directories for each image
            dataset_split (float): from 0 to 1, persent of dataset items to put into the training folder

        Returns:
            (train_path, val_path) Path to training and validation folders
    """

    project_dir = features_path.parent
    train_path = project_dir / "train"
    val_path = project_dir / "val"

    # create directories if don't exist
    for path in [train_path, val_path]:
        if not path.exists():
            path.mkdir()

    train_features_dir = train_path / "image"
    train_labels_dir = train_path / "label"
    val_features_dir = val_path / "image"
    val_labels_dir = val_path / "label"

    # create directories if don't exist
    for path in [train_features_dir, train_labels_dir, val_features_dir, val_labels_dir]:
        if not path.exists():
            path.mkdir()
        else:
            # delete all files from the directory
            for file in path.iterdir():
                file.unlink(missing_ok=True) # FileNotFoundError exceptions will be ignored

    # all ids in the dataset
    dataset_item_ids = labels_path.iterdir()
    dataset_split_idx = int(len(dataset_item_ids) * dataset_split)

    train_dataset_item_ids = dataset_item_ids[:dataset_split_idx]
    val_dataset_item_ids = dataset_item_ids[dataset_split_idx:]


    for item_id in features_path.iterdir():
        
        # load feature image
        image = cv2.imread(features_path / item_id)

        # split feature image
        split_image_on_tiles(image_label, item_id, tile_size, tiles_extension)


    # split all labels  
    for item_id in train_dataset_item_ids:

        # load all vector files for each class
        # rasterize them and unite into one multichannel image and split  
        image_label = []
        label_dirpath = labels_path / item_id
        for label_filename in label_dirpath.iterdir():
            im = rasterize_image(label_dirpath / label_image); # specific class rasterized image
            image_label.append(im)
        image_label = np.asarray(image_label)

        # split label
        split_image_on_tiles(image_label, item_id, tile_size, tiles_extension)


        label_image = rasterize_image()
        split_image_on_tiles(label_image)

        

    # split dataset into train and val





    pass








