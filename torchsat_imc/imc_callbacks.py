
# embedded module defined in c++ code
import sys
from pathlib import Path

try:
    import imc_api
except ImportError:
    import imc_api_cli as imc_api


def confirm_running(training_panel: imc_api.TrainingPanelPrt):
    """ confirm training started """
    if training_panel:
        imc_api.confirm_running(training_panel)


def show_message(training_panel: imc_api.TrainingPanelPrt, title: imc_api.MessageTitle, message: str, message_to_log: str = ""):
    """ show message to user """
    if training_panel:
        imc_api.show_message(training_panel, title, message, message_to_log)
    else:
        print(title.name, message, message_to_log)


def log_message(training_panel: imc_api.TrainingPanelPrt, title: imc_api.MessageTitle, message: str):
    """ log message """
    if training_panel:
        imc_api.log_message(training_panel, title, message)
    else:
        print(title.name, message)


def update_progress(dProgressCounter: float, progress_bar_title: str, progress_bar: imc_api.ProgressBarPtr):
    """ update progress bar """
    if progress_bar:
        imc_api.update_progress(dProgressCounter, progress_bar_title, progress_bar)


def update_preview_image(preview_update_params: imc_api.UpdatePreviewParams, training_panel: imc_api.TrainingPanelPrt):
    """ update preview image to show training progress """
    if training_panel and preview_update_params:
        imc_api.update_preview_image(preview_update_params, training_panel)


def add_checkpoint(training_checkpoint: imc_api.SegmentationModelCheckpoint, training_panel: imc_api.TrainingPanelPrt):
    """ add checkpoint to list """
    if training_panel:
        imc_api.add_checkpoint(training_checkpoint, training_panel)


def check_progress_bar_cancelled(progress_bar: imc_api.ProgressBarPtr) -> bool:
    """ check progress bar status if its cancelled """
    if progress_bar:
        return imc_api.check_progress_bar_cancelled(progress_bar)
