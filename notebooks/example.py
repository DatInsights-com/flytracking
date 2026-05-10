# %%
from fly_tracking import *
import time

OUT_DIR = "./output"
OVERWRITE = False
GENERATE_VIDEOS = True
NB_BG_FRAMES = int(1000)
TOTAL_FRAMES = int(18000)
MAX_HISTORY = int(10)

VIDEO_PATH = "./videos/THshi_2_butanone_260925_3_gd4_15.mp4"

# %%
if __name__ == "__main__":

    print("generating background ... ", end="", flush=True)
    start_time = time.time()
    create_background_image(VIDEO_PATH, OUT_DIR, NB_BG_FRAMES, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    print("detecting flies ... ", end="", flush=True)
    start_time = time.time()
    detect_moving_objects(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    print("tracking flies ... ", end="", flush=True)
    start_time = time.time()
    track_moving_objects(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    print("connecting tracklets ... ", end="", flush=True)
    start_time = time.time()
    connect_tracklets(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    print("generating summary plots ... ", end="", flush=True)
    start_time = time.time()
    plot_longtracks_summary(VIDEO_PATH, OUT_DIR, OVERWRITE)
    print("%s s" % int(time.time() - start_time))

    if GENERATE_VIDEOS:
        print("generating detection video ... ", end="", flush=True)
        start_time = time.time()
        save_detection_movie(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, OVERWRITE)
        print("%s s" % int(time.time() - start_time))

        print("generating tracking video ... ", end="", flush=True)
        start_time = time.time()
        save_tracking_movie(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, MAX_HISTORY, OVERWRITE)
        print("%s s" % int(time.time() - start_time))

        print("generating long tracks video ... ", end="", flush=True)
        start_time = time.time()
        save_longtracks_movie(VIDEO_PATH, OUT_DIR, TOTAL_FRAMES, MAX_HISTORY, OVERWRITE)
        print("%s s" % int(time.time() - start_time))
