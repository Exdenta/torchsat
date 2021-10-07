"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-08-30 11:00:00
 * @modify date 2021-08-30 11:00:00
 * @desc run segmentation model on image script
"""

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

import cv2
import torch
import argparse
import rasterio
import numpy as np
from osgeo import gdal
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from rasterio.windows import Window
from torchsat_imc.models.utils import get_model
from torchsat_imc.transforms import functional
import torchsat_imc.imc_callbacks as imc_callbacks
import torchsat_imc.transforms.transforms_seg as T_seg


# def process_image(  model, image_path: Path, preview_outdir: Path, channel_count: int, tile_size: int, device: str = 'cpu', drop_last: bool = False, 
#                     mean: np.ndarray = np.array([0.302,0.280,0.241]), std: np.ndarray = np.array([0.215,0.194,0.176]), 
#                     training_panel: imc_api.TrainingPanelPrt = None, progress_bar: imc_api.ProgressBarPtr = None, current_progress: float = 0.0) -> bool:
#     """Process image with segmentation model 
    
#     Args:
#         model: loaded pytorch model
#         image_path (Path): full path to the image to process
#         preview_outdir (Path): full path to the preview output directory
#         channel_count (int): input model channel count
#         tile_size (int): tile size to split image into (also model input size)
#         device (str): 'cpu' of 'gpu', device to run model on
#         drop_last (bool): drop last tiles when splitting on tiles or not
#         training_panel: ptr to training panel for callbacks
#         progress_bar: ptr to progress bar dialog for callbacks
#         current_progress (float): current progress bar value

#     Returns:
#         (bool) if operation was successfull
#     """

#     # load image
#     if not image_path.is_file():
#         imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("file {} does not exits.".format(image_path)))
#         return False

#     # split image and label
#     img_src = rasterio.open(image_path)
#     if img_src.count < channel_count:
#         imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("image must have at least {} channels!".format(channel_count)))
#         return False
    
#     # check image has the same number of channels as std and mean vectors
#     if img_src.count != mean.shape[0]:
#         imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("Number of image channels must be the same as the size of mean and std!"))
#         return False

#     # check channel count 
#     if mean.shape != std.shape:
#         imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("Std and mean arrays must be the same size!"))
#         return False

#     rows = img_src.height // tile_size if drop_last else img_src.height // tile_size + 1
#     cols = img_src.width  // tile_size if drop_last else img_src.width  // tile_size + 1

#     if progress_bar:
#         progress_step = (100.0 - current_progress) / (cols * rows + 1)
#         current_progress += progress_step
#         imc_callbacks.update_progress(current_progress, _("Processing image"), progress_bar)

#     model.eval()
#     with torch.no_grad():
#         #
#         # process first tile for info to reserve memory for all tiles
#         #

#         softmax = nn.Softmax(dim=0)

#         def process(arr: np.ndarray) -> np.array:
#             input = functional.to_tensor(arr).to(device)
#             input = input.permute((1, 2, 0))
#             input = functional.normalize(input, mean, std)
#             input = input.unsqueeze(0)
#             return softmax(model(input).squeeze(0)).cpu()

#         first_processed_tile = process(img_src.read(window=Window(0, 0, tile_size, tile_size), boundless=True))
#         processed_image = np.zeros((first_processed_tile.shape[0], first_processed_tile.shape[1] * rows, first_processed_tile.shape[2] * cols)) # reserve memory
#         processed_image[:, :tile_size, :tile_size] = first_processed_tile

#         #
#         # process tiles
#         #
    
#         for row in range(0, rows):
#             for col in range(0, cols):

#                 # skip first tile
#                 if col == 0 and row == 0:
#                     continue

#                 patched_arr = img_src.read(window=Window(col * tile_size, row * tile_size, tile_size, tile_size), boundless=True)
#                 processed_image[:, col * tile_size: (col + 1) * tile_size, row * tile_size: (row + 1) * tile_size] = process(patched_arr[:channel_count])
                
#                 # update progress
#                 if progress_bar:
#                     current_progress += progress_step
#                     imc_callbacks.update_progress(current_progress, _("Processing image"), progress_bar)
#                     if imc_callbacks.check_progress_bar_cancelled(progress_bar):
#                         return False

#         try:
#             if processed_image != None:    
#                 output_filename = f"preview_cls_{image_path.stem}"
#                 filepath = preview_outdir / output_filename + ".tif"
#                 imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogProcessingError, "Preview image shape: " + str(processed_image.shape) + " dtype: " + str(image.dtype))
#                 with rasterio.open( filepath, 'w',
#                                     driver='GTiff',
#                                     count=processed_image.shape[0],
#                                     height=processed_image.shape[1],
#                                     width=processed_image.shape[2],
#                                     dtype=processed_image.dtype) as dst:
#                     dst.write(processed_image)

#                 imc_callbacks.update_preview_image(imc_api.UpdatePreviewParams(filepath, output_filename), training_panel)
#             else:
#                 imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogProcessingError, _("Failed to process the preview image. Model training will continue"))

#         except Exception as e:
#             imc_api.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
#             return False


#     return True


def get_image_transformation(image_src: rasterio.DatasetReader, device):
    """ create transformations for the image
    """
    image = image_src.read()
    # c = np.hstack(image) 
    # mean, std = np.mean(c), np.std(c)
    
    # calculate mean and std for each channel
    mean_std = [cv2.meanStdDev(x) for x in image]
    mean = [x[0][0][0] for x in mean_std]
    std = [x[1][0][0] for x in mean_std]

    image_transform = T_seg.Compose([
        T_seg.ToTensor(),
        T_seg.Normalize(mean, std),
    ])

    # def process(arr: np.ndarray) -> np.ndarray:
    #     input = functional.to_tensor(arr).to(device)
    #     input = input.permute((1, 2, 0))
    #     input = functional.normalize(input, mean, std)
    #     return input

    return image_transform

def process_image(  model, image_path: Path, 
                    channel_count: int, 
                    tile_size: int, 
                    device: str = 'cpu', 
                    drop_last: bool = False, 
                    training_panel: imc_api.TrainingPanelPrt = None, 
                    progress_bar: imc_api.ProgressBarPtr = None, 
                    current_progress: float = 0.0) -> np.ndarray:
    """Process image with segmentation model 
    
    Args:
        model: loaded pytorch model
        image_path (Path): full path to the image to process
        preview_outdir (Path): full path to the preview output directory
        channel_count (int): input model channel count
        tile_size (int): tile size to split image into (also model input size)
        device (str): 'cpu' of 'gpu', device to run model on
        drop_last (bool): drop last tiles when splitting on tiles or not
        training_panel: ptr to training panel for callbacks
        progress_bar: ptr to progress bar dialog for callbacks
        current_progress (float): current progress bar value

    Returns:
        image (np.ndarray)
    """

    # load image
    if not image_path.is_file():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("file {} does not exits.".format(image_path)))
        return None

    # split image and label
    img_src = rasterio.open(image_path)
    if img_src.count < channel_count:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("image must have at least {} channels!".format(channel_count)))
        return None
    
    # calculate image mean and std
    transform = get_image_transformation(img_src, device)

    rows = img_src.height // tile_size if drop_last else img_src.height // tile_size + 1
    cols = img_src.width  // tile_size if drop_last else img_src.width  // tile_size + 1

    if progress_bar:
        progress_step = (100.0 - current_progress) / (cols * rows + 1)
        current_progress += progress_step
        imc_callbacks.update_progress(current_progress, _("Processing image"), progress_bar)

    model.eval()
    with torch.no_grad():
        softmax = nn.Softmax(dim=0)

        # def process(arr: np.ndarray) -> np.array:
        #     input = functional.to_tensor(arr).to(device)
        #     input = input.permute((1, 2, 0))
        #     input = functional.normalize(input, mean, std)
        #     input = input.unsqueeze(0)
        #     return softmax(model(input).squeeze(0)).cpu()
        
        #
        # process first tile for info to reserve memory for all tiles
        #

        mask = np.array((channel_count, tile_size, tile_size)) # TODO: get rid of this useless mask
        tile = img_src.read(window=Window(0, 0, tile_size, tile_size), boundless=True)
        tile = np.transpose(tile, axes=[1, 2, 0])
        tile, mask = transform(tile, mask)
        tile = softmax(tile).cpu().detach().numpy()

        processed_image = np.zeros((tile.shape[0], tile.shape[1] * rows, tile.shape[2] * cols)) # reserve memory
        processed_image[:, :tile_size, :tile_size] = tile

        #
        # process tiles
        #
    
        for row in range(0, rows):
            for col in range(0, cols):

                # skip first tile
                if col == 0 and row == 0:
                    continue
                
                # processs tile
                tile = img_src.read(window=Window(col * tile_size, row * tile_size, tile_size, tile_size), boundless=True)
                tile = np.transpose(tile, axes=[1, 2, 0])
                tile, mask = transform(tile, mask)
                tile = softmax(tile).cpu().detach().numpy()

                processed_image[:, row * tile_size: (row + 1) * tile_size, col * tile_size: (col + 1) * tile_size] = tile[:channel_count]
                
                # update progress
                if progress_bar:
                    current_progress += progress_step
                    imc_callbacks.update_progress(current_progress, _("Processing image"), progress_bar)
                    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
                        return None

    return processed_image


def run_segmentation(params: imc_api.SegmentationInferenceParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> bool:
    """ Segmentation model inference on image
    Args:
        params (imc_api.SegmentationInferenceParams): params for inference
        training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks
        progress_bar (imc_api.ProgressBarPtr): progress bar ptr for progress updates
    """

    # set up hardware to run on
    if params.device == 'cuda' and not torch.cuda.is_available():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("CUDA is not available. Falling back to CPU"))   
        params.device = 'cpu'
    params.device = torch.device('cuda' if params.device == 'cuda' else 'cpu')
    torch.cuda.empty_cache()

    # progress update for progress bar
    current_progress = 0.0
    progress_step = 1.0
    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Loading model..."), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return False

    # load model
    model = get_model(params.model_arch, params.num_classes, pretrained=False)
    model.load_state_dict(torch.load(params.model_path, map_location=params.device))
    model.to(params.device)
    model.eval()

    # process image
    try:
        image_path = Path(params.image_path)
        preview_outdir = Path(params.preview_outdir)

        image = process_image(model, 
                             image_path=image_path,
                             channel_count=params.channel_count, 
                             tile_size=params.tile_size, 
                             current_progress=current_progress, 
                             device=params.device,
                             drop_last=False,
                             training_panel=training_panel, 
                             progress_bar=progress_bar)

        try:
            if image is None:
                return False

            output_filename = f"preview_cls_{image_path.stem}"
            filepath = preview_outdir / Path(output_filename + ".tif")
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogProcessingError, "Preview image shape: " + str(image.shape) + " dtype: " + str(image.dtype))
            image = (image * 255).astype(rasterio.uint8)
            with rasterio.open( filepath, 'w', driver='GTiff',
                                count=image.shape[0],
                                height=image.shape[1],
                                width=image.shape[2],
                                dtype=image.dtype,
                                # transform=rasterio.Affine.identity
                                ) as dst:
                dst.write(image)

                imc_callbacks.update_preview_image(imc_api.UpdatePreviewParams(filepath, output_filename), training_panel)

        except Exception as e:
            imc_api.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
            return False

    except MemoryError as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e) + ". Trying using less memory-hungry algorithm...")
    except Exception as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
    
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='path to an image', required=True)
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', required=True)
    parser.add_argument('--preview_outdir', type=str, help='output directory for processed image', required=True)
    parser.add_argument('--model_arch', type=str, default="unet34", help='model architecture name', required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--channel_count', type=int, required=True)
    parser.add_argument('--tile_size', type=int, required=True)
    parser.add_argument('--device', type=str, default="cuda", required=True)
    args = parser.parse_args()

    params = imc_api.SegmentationInferenceParams(args.image_path, args.model_path, args.preview_outdir, args.model_arch, 
                                                 args.num_classes, args.channel_count, args.tile_size, args.device)

    image = run_segmentation(params, training_panel=None, progress_bar=None)
    if image == None:
        print("Processing has failed")

# def split_image_on_tiles(training_panel: imc_api.TrainingPanelPrt, image_filepath: Path, tile_size: int, channel_count: int, drop_last: bool) -> np.array:
#     """ Split image on tiles

#         Args:
#             image_filepath (Path): full path to image file
#             tile_size (int): tile size
#             channel_count (int): channel count to take from each image
#             drop_last (bool): drop last tiles in the edges of the image
#     """

#     if not image_filepath.is_file():
#         imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("file {} does not exits.".format(image_filepath)))
#         return False


#     # split image and label
#     img_src = rasterio.open(image_filepath)
#     rows = img_src.meta['height'] // tile_size if drop_last else img_src.meta['height'] // tile_size + 1
#     cols = img_src.meta['width']  // tile_size if drop_last else img_src.meta['width']  // tile_size + 1
#     channels = img_src.meta['channels']


#     # allocate memory for tiles (tiles num, channel num, tile width, tile height)
#     image_tiles = np.zeros((rows * cols, channels, tile_size, tile_size))

#     for row in range(rows):
#         for col in range(cols):
#             idx = row * cols + col
#             patched_arr = img_src.read(window=Window(col * tile_size, row * tile_size, tile_size, tile_size), boundless=True)
#             image_tiles[idx] = patched_arr[:channel_count] # take first N channels from image

#     return image_tiles
