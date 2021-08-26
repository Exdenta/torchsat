
# embedded module defined in c++ code
import imc_api

def confirm_running(training_panel: imc_api.TrainingPanelPrt):
    imc_api.confirm_running(training_panel)

def stop_training(training_panel: imc_api.TrainingPanelPrt):
    imc_api.stop_training(training_panel)

def update_epoch(epoch: int, training_panel: imc_api.TrainingPanelPrt):
    imc_api.update_epoch(epoch, training_panel)

def show_message(title: imc_api.MessageTitle, message: str, message_to_log: str = ""):
    imc_api.show_message(title, message, message_to_log)
    
def log_message(title: imc_api.MessageTitle, message: str):
    imc_api.log_message(title, message)

def update_progress(dProgressCounter: float, progress_bar: imc_api.ProgressBarPtr):
    imc_api.update_progress(dProgressCounter, progress_bar)

def add_checkpoint(training_checkpoint: imc_api.TrainingCheckpoint, training_panel: imc_api.TrainingPanelPrt):
    imc_api.add_checkpoint(training_checkpoint, training_panel)