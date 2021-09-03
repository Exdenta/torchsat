"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-08-30 11:00:00
 * @modify date 2021-08-30 11:00:00
 * @desc run segmentation model on image script
"""

from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np
from pathlib import Path
from rasterio.windows import Window
from torchsat_imc.models.utils import get_model
import torchsat_imc.imc_callbacks as imc_callbacks

import gettext
_ = gettext.gettext
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
try:
    import imc_api                     
except ImportError:
    import imc_api_cli as imc_api


def process_image(model, image_path: Path, channel_count: int, tile_size: int, device: str = 'cpu', drop_last: bool = False, training_panel: imc_api.TrainingPanelPrt = None, progress_bar: imc_api.ProgressBarPtr = None, current_progress: float = 0.0) -> bool:
    """Process image with segmentation model 
    
    Args:
        model: loaded pytorch model
        image_path (Path): full path to the image to process
        channel_count (int): input model channel count
        tile_size (int): tile size to split image into (also model input size)
        device (str): 'cpu' of 'gpu', device to run model on
        drop_last (bool): drop last tiles when splitting on tiles or not
        training_panel: ptr to training panel for callbacks
        progress_bar: ptr to progress bar dialog for callbacks
        current_progress (float): current progress bar value

    Returns:
        image (np.array): model output in CHW format  
    """

    # load image
    if not image_path.is_file():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("file {} does not exits.".format(image_path)))
        return False

    # split image and label
    img_src = rasterio.open(image_path)
    if img_src.count < channel_count:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _("image must have at least {} channels!".format(channel_count)))
        return False
    
    rows = img_src.height // tile_size if drop_last else img_src.height // tile_size + 1
    cols = img_src.width  // tile_size if drop_last else img_src.width  // tile_size + 1

    if progress_bar:
        progress_step = (100.0 - current_progress) / (cols * rows + 1)
        current_progress += progress_step
        imc_callbacks.update_progress(current_progress, _("Processing image"), progress_bar)

    # process tiles
    processed_image = np.zeros((img_src.count, rows, cols))
    softmax = nn.Softmax(dim=0)
    with torch.no_grad():
        for row in range(rows):
            for col in range(cols):
                patched_arr = img_src.read(window=Window(col * tile_size, row * tile_size, tile_size, tile_size), boundless=True)
                input = patched_arr[:channel_count].to(device)
                processed_image[:, col * tile_size: (col + 1) * tile_size, row * tile_size: (row + 1) * tile_size] = softmax(model(input))
        
                # update progress
                if progress_bar:
                    current_progress += progress_step
                    imc_callbacks.update_progress(current_progress, _("Processing image"), progress_bar)
                    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
                        return True

    return processed_image


def run_segmentation(params: imc_api.InferenceParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> np.array:
    """ Segmentation model inference on image
    Args:
        params (imc_api.InferenceParams): params for inference
        training_panel (imc_api.TrainingPanelPrt): training panel ptr for callbacks
        progress_bar (imc_api.ProgressBarPtr): progress bar ptr for progress updates
    """

    # set up hardware to run on
    if params.device == imc_api.Device.CUDA and not torch.cuda.is_available():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo, _("CUDA is not available. Falling back to CPU"))   
        device = imc_api.Device.CPU
    params.device = torch.device('cuda' if device == imc_api.Device.CUDA else 'cpu')
    torch.cuda.empty_cache()

    # progress update for progress bar
    current_progress = 0.0
    progress_step = 1.0
    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Loading model..."), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return None

    # load model
    model = get_model(params.model_arch, params.num_classes, pretrained=False)
    model.load_state_dict(torch.load(params.model_path, map_location=params.device))
    model.to(params.device)
    model.eval()

    # process image
    try:
        return process_image(model, params.image_path, params.channel_count, params.tile_size, current_progress, params.device, False, training_panel, progress_bar)
    except MemoryError as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e) + ". Trying using less memory-hungry algorithm...")
    except Exception as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
    
    return None





if __name__ == "__main__":
    args = ArgumentParser()
    image = run_segmentation(args.params, training_panel=None, progress_bar=None)
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
