"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-08-30 11:00:00
 * @modify date 2021-08-30 11:00:00
 * @desc contains some useful dataset api like image splitting and image rasterization
"""


import os
import cv2
import numpy as np
from PIL import Image
from itertools import product
from pathlib import Path
from torchsat_imc.scripts.make_mask_seg_onehot import split_image_and_label
from torchsat_imc.scripts.make_mask_cls import make_mask_cls

# def make_mask_seg(image_filepath: Path, label_dirpath: Path, label_classes: set, tile_size: int, drop_last: bool, image_outdir: Path, label_outdir: Path, id_separator: str, tile_ext: str) -> bool:
#     """ Rasterize vector geojson files into one raster image, 
#         split image and rasterized label into tiles
#     """


#     pass


def delete_item(item_id: str, splitted_image_dirpath: Path, splitted_labels_dirpath: Path, id_separator: str):
    """ Deletes item from the filesystem
    
        Args:
            item_id (str): item id to remove
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
        if image_id == item_id:
            Path(splitted_image_dirpath / image_filename).unlink(True)

    # delete label tiles
    for label_filename in os.listdir(splitted_labels_dirpath):
        label_parts = label_filename.split(id_separator)
        if len(label_parts) <= 0:
            continue
        label_id = label_parts[0]
        if label_id == item_id:
            Path(splitted_labels_dirpath / label_filename).unlink(True)


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



# def split_image(image: np.array, tile_size: int, out_dir: Path, image_name: str, id_separator: str, out_ext: str):
#     """ Split image on tiles
    
#         Args:
#             image (np.array) image matrix
#             tile_size (int) tile size
#             out_dir (Path) output directory for splitted images
#             image_name (str) image name for saving tiles
#             out_ext (str): output tiles extension
#     """
#     image_height = image.shape[0]
#     image_width = image.shape[1]
#     for r in range(0, image_height, tile_size):
#         for c in range(0, image_width, tile_size):
#             cv2.imwrite(f"{str(out_dir / image_name)}{id_separator}{r}_{c}.{out_ext}", image[r:r+tile_size, c:c+tile_size,:])


# def split_images(item_ids: set, images_path: Path, crop_size: int, id_separator: str, out_dir: Path, out_ext: str):
#     """ Rasterize labels and save them as images

#         Args:
#             item_ids (set): image names to take from images_path
#             images_path (Path): path to the directory with images to split
#             crop_size (int): tile size
#             id_separator (str): separator for i
#             out_dir (Path): pth

#             items_ids (set): item ids to rasterize
#             labels_path (Path): path to vector label files
#             out_dir (Path): output directory for rasterized labels
#             out_ext (str): extension for rasterized labels
#     """



#     image_height = image.shape[0]
#     image_width = image.shape[1]
#     for r in range(0, image_height, tile_size):
#         for c in range(0, image_width, tile_size):
#             cv2.imwrite(f"{image_name}_{r}_{c}.{output_extension}", image[r:r+tile_size, c:c+tile_size,:])

#     pass


# def split_dataset(features_path: Path, labels_path: Path, dataset_split: float, tile_size: int, tiles_extension: str) -> tuple(Path, Path):
#     """ Splits dataset into training and validation directories for SegmentationDataset
#             - rasterizes labels
#             - splits images on tiles

#         Folder structure after splitting:

#         ├── 'train_path'
#         │   ├── image
#         │   │   ├── train_1.png
#         │   │   ├── train_2.png
#         │   │   ├── ...
#         │   └── label
#         │       ├── train_1.png
#         │       ├── train_2.png
#         │       ├── ...
#         └── 'val_path'
#             ├── image
#             │   ├── val_10.png
#             │   ├── val_11.png
#             │   ├── ...
#             └── label
#                 ├── val_10.png
#                 ├── val_11.png
#                 ├── ...

#         Args:
#             features_path (pathlib.Path): path to directory with feature images
#             labels_path (pathlib.Path): path to directory with other labels directories for each image
#             dataset_split (float): from 0 to 1, persent of dataset items to put into the training folder

#         Returns:
#             (train_path, val_path) Path to training and validation folders
#     """

#     project_dir = features_path.parent
#     train_path = project_dir / "train"
#     val_path = project_dir / "val"

#     # create directories if don't exist
#     for path in [train_path, val_path]:
#         if not path.exists():
#             path.mkdir()

#     train_features_dir = train_path / "image"
#     train_labels_dir = train_path / "label"
#     val_features_dir = val_path / "image"
#     val_labels_dir = val_path / "label"

#     # create directories if don't exist
#     for path in [train_features_dir, train_labels_dir, val_features_dir, val_labels_dir]:
#         if not path.exists():
#             path.mkdir()
#         else:
#             # delete all files from the directory
#             for file in path.iterdir():
#                 file.unlink(missing_ok=True) # FileNotFoundError exceptions will be ignored

#     # all ids in the dataset
#     dataset_item_ids = labels_path.iterdir()
#     dataset_split_idx = int(len(dataset_item_ids) * dataset_split)

#     train_dataset_item_ids = dataset_item_ids[:dataset_split_idx]
#     val_dataset_item_ids = dataset_item_ids[dataset_split_idx:]


#     for item_id in features_path.iterdir():
        
#         # load feature image
#         image = cv2.imread(features_path / item_id)

#         # split feature image
#         split_image_on_tiles(image_label, item_id, tile_size, tiles_extension)


#     # split all labels  
#     for item_id in train_dataset_item_ids:

#         # load all vector files for each class
#         # rasterize them and unite into one multichannel image and split  
#         image_label = []
#         label_dirpath = labels_path / item_id
#         for label_filename in label_dirpath.iterdir():
#             im = rasterize_image(label_dirpath / label_image); # specific class rasterized image
#             image_label.append(im)
#         image_label = np.asarray(image_label)

#         # split label
#         split_image_on_tiles(image_label, item_id, tile_size, tiles_extension)


#         label_image = rasterize_image()
#         split_image_on_tiles(label_image)

        

#     # split dataset into train and val





#     pass








