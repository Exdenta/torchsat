
# embedded module defined in c++ code
import sys

try:
    import imc_api                     
except ImportError:
    import imc_api_cli as imc_api

def confirm_running(training_panel: imc_api.TrainingPanelPrt):
    imc_api.confirm_running(training_panel)

def stop_training(training_panel: imc_api.TrainingPanelPrt):
    imc_api.stop_training(training_panel)

def update_epoch(epoch: int, training_panel: imc_api.TrainingPanelPrt):
    imc_api.update_epoch(epoch, training_panel)

def show_message(training_panel: imc_api.TrainingPanelPrt, title: imc_api.MessageTitle, message: str, message_to_log: str = ""):
    imc_api.show_message(training_panel, title, message, message_to_log)
    
def log_message(training_panel: imc_api.TrainingPanelPrt, title: imc_api.MessageTitle, message: str):
    imc_api.log_message(training_panel, title, message)

def update_progress(dProgressCounter: float, progress_bar_title: str, progress_bar: imc_api.ProgressBarPtr):
    imc_api.update_progress(dProgressCounter, progress_bar_title, progress_bar)

def add_checkpoint(training_checkpoint: imc_api.SegmentationModelCheckpoint, training_panel: imc_api.TrainingPanelPrt):
    imc_api.add_checkpoint(training_checkpoint, training_panel)

def check_progress_bar_cancelled(progress_bar: imc_api.ProgressBarPtr) -> bool:
    """ check progress bar status if its cancelled """
    return imc_api.check_progress_bar_cancelled(progress_bar)
