"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 30-08-2021 11:00:00
 * @modify date 30-08-2021 11:00:00
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


def get_image_transformation(mean: list, std: list):
    """ create transformations for the image
    """
    image_transform = T_seg.Compose([
        T_seg.ToTensor(),
        T_seg.Normalize(mean, std),
    ])
    
    return image_transform


def process_image(model, image_path: Path,
                  channel_count: int,
                  tile_size: int,
                  mean: list, 
                  std: list,
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
        mean (list of floats): mean values for image normalization model was trained with
        std (list of floats): std values for image normalization model was trained with
        device (str): 'cpu' of 'gpu', device to run model on
        drop_last (bool): drop last tiles when splitting on tiles or not
        training_panel: ptr to training panel for callbacks
        progress_bar: ptr to progress bar dialog for callbacks
        current_progress (float): current progress bar value

    Returns:
        image (np.ndarray): processed image
    """

    # load image
    if not image_path.is_file():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError,
                                   _("file {} does not exits.".format(image_path)))
        return None

    # split image and label
    img_src = rasterio.open(image_path)
    if img_src.count < channel_count:
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogError, _(
            "image must have at least {} channels!".format(channel_count)))
        return None

    # calculate image mean and std
    transform = get_image_transformation(mean, std)

    rows = img_src.height // tile_size if drop_last else img_src.height // tile_size + 1
    cols = img_src.width // tile_size if drop_last else img_src.width // tile_size + 1

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

        mask = np.array((channel_count, tile_size, tile_size))  # TODO: get rid of this useless mask
        tile = img_src.read(window=Window(0, 0, tile_size, tile_size), boundless=True)
        tile = np.transpose(tile, axes=[1, 2, 0])
        tile, mask = transform(tile, mask)
        tile = softmax(tile).cpu().detach().numpy()

        processed_image = np.zeros((tile.shape[0], tile.shape[1] * rows, tile.shape[2] * cols))  # reserve memory
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
                tile = img_src.read(window=Window(col * tile_size, row * tile_size,
                                    tile_size, tile_size), boundless=True)
                tile = np.transpose(tile, axes=[1, 2, 0])
                tile, mask = transform(tile, mask)
                tile = softmax(tile).cpu().detach().numpy()

                processed_image[:, row * tile_size: (row + 1) * tile_size, col *
                                tile_size: (col + 1) * tile_size] = tile[:channel_count]

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
    if params.device == imc_api.Device.CUDA and not torch.cuda.is_available():
        imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInfo,
                                   _("CUDA is not available. Falling back to CPU"))
        params.device = imc_api.Device.CPU
    device = torch.device('cuda' if params.device == imc_api.Device.CUDA else 'cpu')
    # params.device = torch.device('cuda' if params.device == imc_api.Device.CUDA else 'cpu')

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # progress update for progress bar
    current_progress = 0.0
    progress_step = 1.0
    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Loading model..."), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return False

    # check model checkpoint path, if no extension - add pytorch model extension
    if not params.model_path.is_file():
        found_model = False
        for path in params.model_path.parent.iterdir():
            if path.is_file():
                if path.stem == params.model_path.stem:
                    # found file with stem = model name
                    params.model_path = path
                    found_model = True
                    break
        if not found_model:
            imc_callbacks.show_message(training_panel, imc_api.MessageTitle.LogInitError,
                                       _("Wrong checkpoint path. File not found!"))
            return False

    # load model
    model = get_model(params.model_arch, params.num_classes, pretrained=False)
    model.load_state_dict(torch.load(params.model_path, map_location=device))
    model.to(device)
    model.eval()

    # process image
    try:
        image_path = Path(params.image_path)
        preview_outdir = Path(params.preview_outdir)

        image = process_image(model,
                              image_path=image_path,
                              channel_count=params.channel_count,
                              tile_size=params.tile_size,
                              mean=params.mean,
                              std=params.std,
                              current_progress=current_progress,
                              device=device,
                              drop_last=False,
                              training_panel=training_panel,
                              progress_bar=progress_bar)

        try:
            if image is None:
                return False

            output_filename = f"preview_cls_{image_path.stem}"
            filepath = preview_outdir / Path(output_filename + ".tif")
            image = (image * 255).astype(rasterio.uint8)
            with rasterio.open(filepath, 'w', driver='GTiff',
                               count=image.shape[0],
                               height=image.shape[1],
                               width=image.shape[2],
                               dtype=image.dtype,
                               # transform=rasterio.Affine.identity
                               ) as dst:
                dst.write(image)

                imc_callbacks.update_preview_image(imc_api.UpdatePreviewParams(
                    filepath, output_filename), training_panel)

        except Exception as e:
            imc_api.log_message(training_panel, imc_api.MessageTitle.LogError, str(e))
            return False

    except MemoryError as e:
        imc_callbacks.log_message(training_panel, imc_api.MessageTitle.LogError, str(e) +
                                  ". Trying using less memory-hungry algorithm...")
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

    device = imc_api.Device.CUDA if args.device == 'cuda' else imc_api.Device.CPU
    params = imc_api.SegmentationInferenceParams(Path(args.image_path), Path(args.model_path), Path(args.preview_outdir), args.model_arch,
                                                 args.num_classes, args.channel_count, args.tile_size, device)

    image = run_segmentation(params, training_panel=None, progress_bar=None)
    if image == None:
        print("Processing has failed")
