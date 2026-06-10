# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: May 7, 2026
# Version: 1.1

import sys
import time
from pathlib import Path
from flytracking import *

OVERWRITE = False
GENERATE_VIDEOS = False
NB_BG_FRAMES = int(1000)
TOTAL_FRAMES = int(18000)
MAX_HISTORY = int(10)


def main():

    if len(sys.argv) != 3:
        print("Usage: python run_fly_tracker.py video_path output_dir")
        sys.exit(1)

    VIDEO_PATH = sys.argv[1]
    OUT_DIR = sys.argv[2]

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(VIDEO_PATH)
    print(OUT_DIR)

    print("generating background ... ", end="", flush=True)
    start_time = time.time()
    create_background_image(VIDEO_PATH, OUT_DIR, NB_BG_FRAMES, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    print("registering background ... ", end="", flush=True)
    start_time = time.time()
    register_background_coordinates(VIDEO_PATH, OUT_DIR, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    print("detecting flies ... ", end="", flush=True)
    start_time = time.time()
    detect_moving_objects(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    if GENERATE_VIDEOS:
        print("generating detection video ... ", end="", flush=True)
        start_time = time.time()
        save_detection_movie(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
        print("%s s" % int(time.time() - start_time))

    print("tracking flies ... ", end="", flush=True)
    start_time = time.time()
    track_moving_objects(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    if GENERATE_VIDEOS:
        print("generating tracking video ... ", end="", flush=True)
        start_time = time.time()
        save_tracking_movie(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, MAX_HISTORY, OVERWRITE)
        print("%s s" % int(time.time() - start_time))

    print("connecting tracklets ... ", end="", flush=True)
    start_time = time.time()
    connect_tracklets(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    if GENERATE_VIDEOS:
        print("generating long tracks video ... ", end="", flush=True)
        start_time = time.time()
        save_longtracks_movie(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, MAX_HISTORY, OVERWRITE)
        print("%s s" % int(time.time() - start_time))

    print("generating summary plots ... ", end="", flush=True)
    start_time = time.time()
    plot_longtracks_summary(VIDEO_PATH, OUT_DIR, OVERWRITE)
    print("%s s" % int(time.time() - start_time))


if __name__ == "__main__":
    main()
