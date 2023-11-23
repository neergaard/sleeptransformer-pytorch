from .download_functions.download_mros import download_mros
from .download_functions.download_shhs import download_shhs
from .download_functions.download_wsc import download_wsc

download_fns = {"mros": download_mros, "wsc": download_wsc, "shhs": download_shhs}
