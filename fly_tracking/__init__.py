# Module for fly tracking in an arena
__author__ = "Yann Dufour"
__version__ = "1.1"

from .fly_detection import *
from .fly_tracking import *
from .connect_tracklets import *
from .processing_tracks import *

__all__ = [
    "create_background_image",
    "register_background_coordinates",
    "tracks_to_dataframe",
    "detect_moving_objects",
    "save_detection_movie",
    "track_moving_objects",
    "save_tracking_movie",
    "connect_tracklets",
    "save_longtracks_movie",
    "load_results_to_df",
    "plot_longtracks_summary",
]
