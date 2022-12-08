import warnings

from lightning.app.storage import Drive
from lightning.app.utilities.cloud import is_running_in_cloud

def warn_if_drive_not_empty(drive: Drive):

    if drive.list():
        warnings.warn("Drive is not empty! This may result into wrong logging behaviour if your app doesn't have a built-in resume mechanism. Consider deleting the .lightning file and restarting the app.")

def warn_if_local():
    if not is_running_in_cloud():
        warnings.warn("This app is optimized for cloud usage. For local testing, please consider choosing a very small model (e.g. bloom-560m) and change the strategy to ddp. Depending on your hardware, you may run out of memory!")