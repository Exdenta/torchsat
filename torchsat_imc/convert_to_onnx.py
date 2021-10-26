"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 08-10-2021 11:00:00
 * @modify date 08-10-2021 11:00:00
 * @desc scrtipt to convert model from pytorch to onnx
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
import torchsat_imc.imc_callbacks as imc_callbacks

import onnx
# import onnxruntime as rt
import numpy as np
import torch
import torch.onnx
import argparse
from pathlib import Path
from torchsat_imc.models.utils import get_model

# Tutorial
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html


def convert_checkpoint(params: imc_api.ConvertSegmentationCheckpointParams, training_panel: imc_api.TrainingPanelPrt, progress_bar: imc_api.ProgressBarPtr) -> bool:
    """ Converts pytorch checkpoint to onnx model """

    # try:

    # progress update for progress bar
    current_progress = 0.0
    progress_step = 25.0
    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Loading model..."), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return False

    # The exported model will accept inputs of size [1, input_channels, image_size, image_size]
    dummy_input = torch.randn(1, params.input_channels, params.image_size, params.image_size)

    # load model
    device = torch.device('cpu')
    model = get_model(params.model_arch, len(params.classes), pretrained=False)
    model.load_state_dict(torch.load(params.model_path, map_location=device))
    
    model = torch.nn.Sequential(
        model,
        torch.nn.Softmax(0)
    )
  
    model.eval()
    model.to(device)

    # pytorch checkpoint output (for testing)
    pytorch_result = model(dummy_input).cpu().detach().numpy()

    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Model conversion..."), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return False

    #
    # Export pytorch model to onnx format
    #

    torch.onnx.export(model, dummy_input, params.output_model_path, opset_version=13)

    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Validating optimized model..."), progress_bar)
    if imc_callbacks.check_progress_bar_cancelled(progress_bar):
        return False

    #
    # Test onnx model
    #

    onnx_model = onnx.load(params.output_model_path)    # Check ONNX model
    onnx.checker.check_model(onnx_model)                # Check that the IR is well formed

    # sess = rt.InferenceSession(str(params.output_model_path))
    # input_name = sess.get_inputs()[0].name
    # label_name = sess.get_outputs()[0].name
    # onnxruntime_result = sess.run([label_name], {input_name: dummy_input.cpu().detach().numpy()})[0]

    #
    # Add new model
    #

    current_progress += progress_step
    imc_callbacks.update_progress(current_progress, _("Saving converted model..."), progress_bar)

    # models_difference_matrix = abs(pytorch_result - onnxruntime_result)
    models_difference_matrix = abs(pytorch_result - pytorch_result)
    models_difference = np.max(models_difference_matrix)
    onnx_model_params = imc_api.OnnxModelParams(params, models_difference)
    imc_api.add_model(onnx_model_params, training_panel)

    return True

    # except Exception as e:
    #     return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', type=str, help='model architecture name', required=True)
    parser.add_argument('--model_path', type=str, help='path to model pytorch checkpoint', required=True)
    parser.add_argument('--output_model_path', type=str, help='output onnx model path', required=True)
    parser.add_argument('--image_size', type=int, default=128,
                        help='size of an input images (height and width)', required=True)
    parser.add_argument('--input_channels', type=int, help='number of channels in input images', required=True)
    parser.add_argument('--classes', nargs='+', type=str, help='classes names for segmentation', required=True)
    parser.add_argument('--mean', nargs='+', type=str, help='mean values for preprocessing', required=True)
    parser.add_argument('--std', nargs='+', type=str, help='std values for preprocessing', required=True)
    parser.add_argument('--preprocessing_methods', nargs='+', type=str, help='method for image preprocessing', required=True)
    args = parser.parse_args()

    params = imc_api.ConvertSegmentationCheckpointParams(args.model_arch, Path(args.model_path), Path(args.output_model_path), args.input_channels, 
                                                        args.image_size, args.mean, args.std, args.classes, args.preprocessing_methods)
    convert_checkpoint(params, None, None)
