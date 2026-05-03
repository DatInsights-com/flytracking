# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: April 17, 2026
# Version: 1.0


import os
from tqdm.notebook import tqdm

import cv2
import numpy as np
import pandas as pd
from math import radians, degrees

from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import stdtr
from scipy.sparse.csgraph import shortest_path

from ultralytics.trackers.byte_tracker import BYTETracker

from itertools import product
from collections import deque

import json
import gzip

import plotnine as pn
import multiprocessing as mp
from time import sleep

import warnings

warnings.filterwarnings("ignore")

# functions necessary to detect an track flies in an arena


# scan files in directories and subdirectories
def scandir_fast(dir: str, ext: str):
    subdirectories, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subdirectories.append(f.path)
        if f.is_file():
            if f.name.endswith(ext):
                files.append(f.path)
    for dir in list(subdirectories):
        sf, f = scandir_fast(dir, ext)
        subdirectories.extend(sf)
        files.extend(f)
    return (subdirectories, files)


# sample frames from movie to generate average background image
def create_background_image(
    video_path: str, out_dir: str, nb_frames: int, max_time: int, overwrite: bool
):
    video_name = os.path.basename(video_path).split(".")[0]

    BACKGROUND_FILE = f"{out_dir}/{video_name}_background.png"

    if os.path.exists(BACKGROUND_FILE) and (not overwrite):
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb_frames = min([nb_frames, total_frames])

    stride = int(np.max([1, max_time * fps / nb_frames]))
    background = np.zeros((height, width), dtype=np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(nb_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
        ret, frame = cap.read()
        if not ret:
            break
        background = cv2.add(
            background,
            cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY),
        )
    cap.release()

    background = cv2.multiply(background, 1.0 / nb_frames)
    background = background.astype(np.uint8)

    cv2.imwrite(BACKGROUND_FILE, background)


# detect moving objects in every frame of movie after background substraction and filtering
def detect_moving_objects(
    video_path: str, out_dir: str, max_time: int, overwrite: bool
):

    video_name = os.path.basename(video_path).split(".")[0]

    DETECTION_FILE = f"{out_dir}/{video_name}_detection.json.gz"
    BACKGROUND_FILE = f"{out_dir}/{video_name}_background.png"

    if os.path.exists(DETECTION_FILE) and (not overwrite):
        return

    MIN_INTESITY = 7
    MIN_AREA = 7
    # MIN_SOLIDITY = 0.7
    # PRCTILE_INTENSITY = 99.9
    MASK_THRESHOLD = 1.9
    GAUSSIAN_STD = 11
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5, 5])
    THREAD_NUM = cv2.getNumberOfCPUs()

    bgframe = cv2.imread(BACKGROUND_FILE, cv2.IMREAD_GRAYSCALE)
    bgframe = bgframe.astype(np.float32)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )

    all_detected_objects = [[] for _ in range(total_frames)]

    pool = mp.Pool(processes=THREAD_NUM)
    pending_tasks = deque()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
        fg_frame = cv2.subtract(bgframe, frame_gray)

        task = pool.apply_async(
            process_frame_detection,
            args=(
                fg_frame,
                GAUSSIAN_STD,
                MASK_THRESHOLD,
                MORPH_KERNEL,
                MIN_INTESITY,
                MIN_AREA,
            ),
        )
        pending_tasks.append(task)
    cap.release()

    idx = 0
    while len(pending_tasks) > 0:
        if pending_tasks[0].ready():
            res = pending_tasks.popleft().get()
            all_detected_objects[idx] = res
            idx += 1
        else:
            sleep(1)

    pool.close()
    pool.join()

    with gzip.open(DETECTION_FILE, "wt") as f:
        json.dump(all_detected_objects, f)


def process_frame_detection(
    fg_frame, gaussian_std, mask_threshold, morph_kernel, min_intensity, min_area
):

    fg_smooth = cv2.subtract(
        cv2.GaussianBlur(fg_frame, (gaussian_std, gaussian_std), 0),
        cv2.GaussianBlur(fg_frame, (gaussian_std * 3, gaussian_std * 3), 0),
    )

    fg_mask = cv2.threshold(fg_smooth, mask_threshold, 255, cv2.THRESH_BINARY)[1]
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kernel).astype(np.uint8)

    total_labels, label_ids = cv2.connectedComponents(fg_mask, 4, cv2.CV_32S)

    for id in range(1, total_labels):
        max_int = np.max(fg_smooth[label_ids == id])
        label_ids[
            (label_ids == id) & (fg_smooth < np.max([min_intensity, (max_int / 3)]))
        ] = 0

    fg_mask = np.astype(255 * (label_ids > 0), np.uint8)
    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours_area = [cv2.contourArea(cnt) for cnt in contours]

    contours, contours_area = zip(
        *[
            (cnt, ar)
            for (cnt, ar) in zip(contours, contours_area)
            if (len(cnt) > 4) and (ar > min_area)
        ]
    )

    objects_ellipses = [cv2.fitEllipse(cnt) for cnt in contours]
    objects_ellipses = [
        obj
        for obj in objects_ellipses
        if (obj[0][0] < fg_mask.shape[1])
        and (obj[0][1] < fg_mask.shape[0])
        and (obj[0][0] >= 0)
        and (obj[0][1] >= 0)
    ]
    # [y, x, axis_long/2, axis_short/2, angle_degree]
    objects_intensities = [
        fg_smooth[int(x), int(y)] for (y, x), (_, _), _ in objects_ellipses
    ]
    objects_properties = np.stack(
        [
            [o[1][0] for o in objects_ellipses],
            [o[1][1] for o in objects_ellipses],
            objects_intensities,
        ],
        axis=1,
    )
    confidence = calculate_confidence(objects_properties)
    objects_ellipses = [
        [a, b, 2.5 * c, 2.5 * d, radians(e), f]
        for ((a, b), (c, d), e), (f) in zip(objects_ellipses, confidence)
        if f > 0
    ]

    return objects_ellipses


# define class to store tracker configuration from ultralytics
class Tracker_cfg:

    def __init__(
        self,
        tracker_type,
        track_high_thresh,
        track_low_thresh,
        new_track_thresh,
        track_buffer,
        match_thresh,
        fuse_score,
    ):
        self.tracker_type = tracker_type
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score


# define class to store detected objects to send to tracker from ultralytics
class Detections:
    def __init__(self, xywhr, conf, obj_cls):
        self.xywhr = xywhr
        self.conf = conf
        self.cls = obj_cls

    def __getitem__(self, index):
        return Detections(self.xywhr[index], self.conf[index], self.cls[index])

    def __len__(self):
        return len(self.conf)


# todo: calculate the confidence that a detected object is a fly
def calculate_confidence(properties):

    ind = np.any(properties <= 0, axis=1)
    properties[ind, :] = 1
    log_prop = np.log(properties)
    mean = np.mean(log_prop[np.invert(ind), :], axis=0)
    cov_matrix = np.cov(log_prop[np.invert(ind), :], rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = [mahalanobis(x, mean, inv_cov_matrix) for x in log_prop]
    confidence = 1 - chi2.cdf(distances, 3)
    confidence[ind] = 0

    return confidence


# track moving objects using bytetracker
def track_moving_objects(video_path: str, out_dir: str, max_time: int, overwrite: bool):

    video_name = os.path.basename(video_path).split(".")[0]

    TRACKING_FILE = f"{out_dir}/{video_name}_tracking.json.gz"
    LEFTOVER_FILE = f"{out_dir}/{video_name}_leftovers.json.gz"
    DETECTION_FILE = f"{out_dir}/{video_name}_detection.json.gz"

    if (
        os.path.exists(TRACKING_FILE)
        and os.path.exists(LEFTOVER_FILE)
        and (not overwrite)
    ):
        return

    TRACKER_TYPE = "bytetrack"
    TRACK_HIGH_THRESH = 0
    TRACK_LOW_THRESH = 0
    NEW_TRACK_THRESH = 0
    TRACK_BUFFER = 0
    MATCH_THRESH = 0.999
    FUSE_SCORE = True

    with gzip.open(DETECTION_FILE, "rt") as f:
        all_detected_objects = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )
    cap.release()

    cfg = Tracker_cfg(
        TRACKER_TYPE,
        TRACK_HIGH_THRESH,
        TRACK_LOW_THRESH,
        NEW_TRACK_THRESH,
        TRACK_BUFFER,
        MATCH_THRESH,
        FUSE_SCORE,
    )
    tracker = BYTETracker(cfg, frame_rate=fps)

    all_tracked_objects = [[] for _ in range(total_frames)]
    not_tracked_objects = [[] for _ in range(total_frames)]

    for i in tqdm(range(total_frames)):

        if not all_tracked_objects[i]:
            objects_ellipses = all_detected_objects[i]
            xywhr = []
            conf = []
            obj_cls = []
            for obj in objects_ellipses:
                xywhr.append(obj[:5])
                conf.append(obj[5]),
                obj_cls.append(0)

            objects = Detections(np.array(xywhr), np.array(conf), np.array(obj_cls))

            tracked_objects = tracker.update(objects)

            # [x, y, w, h, r, track_id, conf, cls, detect_id]
            all_tracked_objects[i] = [obj[:7] for obj in tracked_objects.tolist()]
            dist_objects = cdist(
                np.array([track[:5] for track in all_tracked_objects[i]]),
                np.array([det[:5] for det in all_detected_objects[i]]),
            )
            ind_track, ind_det = linear_sum_assignment(dist_objects)
            not_tracked_objects[i] = [
                all_detected_objects[i][id]
                for id in range(len(all_detected_objects[i]))
                if id not in ind_det
            ]

    with gzip.open(TRACKING_FILE, "wt") as f:
        json.dump(all_tracked_objects, f)

    with gzip.open(LEFTOVER_FILE, "wt") as f:
        json.dump(not_tracked_objects, f)


# generate and save movie annotated with detected objects
def save_detection_movie(video_path: str, out_dir: str, max_time: int, overwrite: bool):

    video_name = os.path.basename(video_path).split(".")[0]

    DETECTION_VIDEO_FILE = f"{out_dir}/{video_name}_detection.mp4"
    DETECTION_FILE = f"{out_dir}/{video_name}_detection.json.gz"

    if os.path.exists(DETECTION_VIDEO_FILE) and (not overwrite):
        return

    with gzip.open(DETECTION_FILE, "rt") as f:
        all_detected_objects = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    detection_movie = cv2.VideoWriter(
        DETECTION_VIDEO_FILE,
        fourcc,
        fps,
        (int(width / 2), int(height / 2)),
    )

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        objects_ellipses = all_detected_objects[i]

        frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_ellipse = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        for obj in objects_ellipses:
            coords = ((obj[0], obj[1]), (obj[2] / 2, obj[3] / 2), degrees(obj[4]))
            frame_ellipse = cv2.ellipse(frame_ellipse, coords, (200, 200, 0), 2)

        frame_ellipse = cv2.resize(
            frame_ellipse,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_LINEAR,
        )
        frame_ellipse = cv2.putText(
            frame_ellipse,
            f"{i:05}" + "/" + f"{total_frames:05}",
            (10, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color=(0, 0, 0),
            thickness=1,
        )
        detection_movie.write(frame_ellipse)

    detection_movie.release()
    cap.release()


def save_leftovers_movie(video_path: str, out_dir: str, max_time: int, overwrite: bool):

    video_name = os.path.basename(video_path).split(".")[0]

    LEFTOVER_VIDEO_FILE = f"{out_dir}/{video_name}_leftovers.mp4"
    LEFTOVER_FILE = f"{out_dir}/{video_name}_leftovers.json.gz"

    if os.path.exists(LEFTOVER_VIDEO_FILE) and (not overwrite):
        return

    with gzip.open(LEFTOVER_FILE, "rt") as f:
        not_tracked_objects = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    detection_movie = cv2.VideoWriter(
        LEFTOVER_VIDEO_FILE,
        fourcc,
        fps,
        (int(width / 2), int(height / 2)),
    )

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        objects_ellipses = not_tracked_objects[i]

        frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_ellipse = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        for obj in objects_ellipses:
            coords = ((obj[0], obj[1]), (obj[2] / 2, obj[3] / 2), degrees(obj[4]))
            frame_ellipse = cv2.ellipse(frame_ellipse, coords, (200, 200, 0), 2)

        frame_ellipse = cv2.resize(
            frame_ellipse,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_LINEAR,
        )
        frame_ellipse = cv2.putText(
            frame_ellipse,
            f"{i:05}" + "/" + f"{total_frames:05}",
            (10, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color=(0, 0, 0),
            thickness=1,
        )
        detection_movie.write(frame_ellipse)

    detection_movie.release()
    cap.release()


# generate and save movie annotated with tracked objects
def save_tracking_movie(
    video_path: str, out_dir: str, max_time: int, max_history: int, overwrite: bool
):

    video_name = os.path.basename(video_path).split(".")[0]

    TRACKING_VIDEO_FILE = f"{out_dir}/{video_name}_tracking.mp4"
    TRACKING_FILE = f"{out_dir}/{video_name}_tracking.json.gz"

    if os.path.exists(TRACKING_VIDEO_FILE) and (not overwrite):
        return

    with gzip.open(TRACKING_FILE, "rt") as f:
        all_tracked_objects = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )

    max_history = int(max_history * fps)
    tracking_history = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tracking_movie = cv2.VideoWriter(
        TRACKING_VIDEO_FILE,
        fourcc,
        fps,
        (int(width / 2), int(height / 2)),
    )

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_tracking = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        tracked_objects = all_tracked_objects[i]

        for track in tracked_objects:
            track_id = track[5]
            if track_id in tracking_history:
                tracking_history[track_id] = np.concatenate(
                    [
                        np.array([[track[:2]]], dtype="int32"),
                        tracking_history[track_id],
                    ],
                    axis=0,
                )
                tracking_history[track_id] = tracking_history[track_id][
                    :max_history, ...
                ]
            else:
                tracking_history[track_id] = np.array([[track[:2]]], dtype="int32")
            coords = (
                (track[0], track[1]),
                (track[2] / 2, track[3] / 2),
                degrees(track[4]),
            )
            color = cv2.applyColorMap(
                np.array(((track_id * 29) % 255), dtype="uint8"), cv2.COLORMAP_HSV
            )[0][0]
            color = tuple(color.tolist())
            frame_tracking = cv2.ellipse(
                frame_tracking,
                coords,
                color,
                2,
            )
            frame_tracking = cv2.polylines(
                frame_tracking,
                [tracking_history[track_id]],
                isClosed=False,
                color=color,
                thickness=2,
            )
            frame_tracking = cv2.putText(
                frame_tracking,
                str(int(track_id)),
                tuple(map(int, np.add(coords[0], 15))),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                color=color,
                thickness=1,
            )

        frame_tracking = cv2.resize(
            frame_tracking,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_LINEAR,
        )
        frame_tracking = cv2.putText(
            frame_tracking,
            f"{i:05}" + "/" + f"{total_frames:05}",
            (10, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color=(0, 0, 0),
            thickness=1,
        )
        tracking_movie.write(frame_tracking)

    tracking_movie.release()
    cap.release()


def connect_tracklets(video_path: str, out_dir: str, max_time: int, overwrite: bool):

    video_name = os.path.basename(video_path).split(".")[0]

    LONG_TRACKS_FILE = f"{out_dir}/{video_name}_longtracks.json.gz"
    TRACKING_FILE = f"{out_dir}/{video_name}_tracking.json.gz"
    LEFTOVER_FILE = f"{out_dir}/{video_name}_leftovers.json.gz"

    if os.path.exists(LONG_TRACKS_FILE) and (not overwrite):
        return

    PROB_NO_DETECTION = 1e-3
    PROB_DOUBLE_DETECTION = 1e-4
    T_DIST_DF = 11
    MIN_PROB_THRESH = 0
    MIN_OBJECT_CONFIDENCE = 0.25

    with gzip.open(TRACKING_FILE, "rt") as f:
        all_tracked_objects = json.load(f)

    with gzip.open(LEFTOVER_FILE, "rt") as f:
        leftover_objects = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )
    cap.release()

    all_tracked_df = linear_interpolate_tracklets(all_tracked_objects)

    estimated_diffusion_coeff = (
        np.nanmean(
            np.square(
                np.linalg.norm(
                    all_tracked_df.groupby("tracklet_id").diff()[["x", "y"]].values,
                    axis=1,
                )
            )
            / all_tracked_df.groupby("tracklet_id").diff()["frame"].values
        )
        / 4
    )

    leftover_df = detection_to_tracklets_df(
        leftover_objects, all_tracked_df["tracklet_id"].max(), MIN_OBJECT_CONFIDENCE
    )
    all_tracked_df = pd.concat([all_tracked_df, leftover_df], ignore_index=True)

    shortest_paths, shortest_costs = find_shortest_paths(
        all_tracked_df,
        estimated_diffusion_coeff,
        PROB_NO_DETECTION,
        PROB_DOUBLE_DETECTION,
        T_DIST_DF,
        MIN_PROB_THRESH,
    )
    all_tracked_relabeled = relabel_tracklets_with_paths(all_tracked_df, shortest_paths)
    all_relabeled_list = [[] for _ in range(total_frames)]
    for frame in range(total_frames):
        all_relabeled_list[frame] = all_tracked_relabeled.loc[
            all_tracked_relabeled["frame"] == frame
        ].values.tolist()

    with gzip.open(LONG_TRACKS_FILE, "wt") as f:
        json.dump(all_relabeled_list, f)


def detection_to_tracklets_df(detected_objects, id, min_confidence):

    list_df = []
    for i, frame in enumerate(detected_objects):
        objects = pd.DataFrame(
            frame,
            columns=[
                "x",
                "y",
                "axis_1",
                "axis_2",
                "orientation",
                "confidence",
            ],
        )
        objects["frame"] = i
        list_df.append(objects)

    objects_df = pd.concat(list_df, ignore_index=True)
    objects_df["frame"] = objects_df["frame"].astype("int")
    objects_df["tracklet_id"] = objects_df.index + id + 1
    objects_df["tracklet_id"] = objects_df["tracklet_id"].astype("int")
    objects_df = objects_df.loc[objects_df["confidence"] >= min_confidence]

    return objects_df


def linear_interpolate_tracklets(all_tracked_objects):

    list_df = []
    for i, frame in enumerate(all_tracked_objects):
        objects = pd.DataFrame(
            frame,
            columns=[
                "x",
                "y",
                "axis_1",
                "axis_2",
                "orientation",
                "tracklet_id",
                "confidence",
            ],
        )
        objects["frame"] = i
        list_df.append(objects)

    objects_df = pd.concat(list_df, ignore_index=True)
    objects_df["frame"] = objects_df["frame"].astype("int")
    objects_df["tracklet_id"] = objects_df["tracklet_id"].astype("int")

    interpolated_df = pd.DataFrame()
    tracklet_id = objects_df["tracklet_id"].unique()
    for id in tracklet_id:
        tracks = objects_df.loc[objects_df["tracklet_id"] == id]
        tracks.index = tracks["frame"]
        if len(tracks) > 1:
            tracks = tracks.reindex(
                np.arange(tracks["frame"].min(), tracks["frame"].max() + 1),
                fill_value=np.nan,
            ).interpolate()
        interpolated_df = pd.concat([interpolated_df, tracks], ignore_index=True)
    interpolated_df["tracklet_id"] = interpolated_df["tracklet_id"].astype("int")
    interpolated_df["frame"] = interpolated_df["frame"].astype("int")
    return interpolated_df


def relabel_tracklets_with_paths(data, paths):
    data_relabeled = data.copy()
    data_relabeled["long_track_id"] = data["tracklet_id"].values

    for track in paths:
        data_relabeled.loc[
            np.isin(data_relabeled["tracklet_id"], track), "long_track_id"
        ] = track[0]

    data_relabeled.sort_values(by=["frame", "tracklet_id"], inplace=True)
    data_relabeled.drop_duplicates(
        ["frame", "long_track_id"], keep="first", ignore_index=True, inplace=True
    )
    return data_relabeled


def find_shortest_paths(
    data, diff_coeff, prob_no_detection, prob_double_detection, df, min_prob
):

    tracklets_id = data["tracklet_id"].unique().tolist()
    tracklets_id.sort()
    tracklets_id.insert(0, 0)
    tracklets_id.append(-1)
    shortest_paths = []
    shortest_costs = []
    tracks = tracklets_id.copy()

    dist_graph = calculate_dist_graph(
        data, diff_coeff, prob_no_detection, prob_double_detection, df
    )

    if min_prob > 0:
        mask = dist_graph > -np.log(min_prob)
        mask[:, -1] = False
        dist_graph[mask] = 0

    while True:
        dist_matrix, predecessors = shortest_path(
            dist_graph, return_predecessors=True, directed=True
        )
        idx1 = 0
        idx2 = -1
        curr_node = idx2  # start from the destination node
        path = []
        nodes = []
        while curr_node != -9999:  # no previous node available, exit the loop
            path = [
                tracks[curr_node]
            ] + path  # prefix the previous node obtained from the last iteration
            nodes = [curr_node] + nodes
            curr_node = int(
                predecessors[idx1, curr_node]
            )  # set current node to previous node
        if len(path) < 3:
            break
        shortest_paths.append(path[1:-1])
        shortest_costs.append(dist_matrix[idx1, idx2])
        dist_graph = np.ascontiguousarray(
            np.delete(np.delete(dist_graph, nodes[1:-1], axis=0), nodes[1:-1], axis=1)
        )
        tracks = [x for x in tracks if x not in path[1:-1]]

    return (shortest_paths, shortest_costs)


def calculate_dist_graph(
    data, diff_coeff, prob_no_detection, prob_double_detection, df
):
    dist_px = calculate_dist_px(data)
    dist_frame = calculate_dist_frame(data)
    dist_confidence = calculate_dist_confidence(data)
    dist_count = calculate_dist_count(data)
    dist_graph = (
        neg_log_p_dist2D(dist_px, diff_coeff, df)
        - np.log(dist_confidence)
        - (np.log(1 - np.power(1 - prob_no_detection, dist_count)))
    )
    dist_graph[dist_frame > 1] = dist_graph[dist_frame > 1] - (
        (dist_frame[dist_frame > 1] - 1) * np.log(prob_no_detection)
    )
    dist_graph[dist_frame < 1] = dist_graph[dist_frame < 1] - (
        -(dist_frame[dist_frame < 1] - 1) * np.log(prob_double_detection)
    )
    dist_graph[:, 0] = 0
    dist_graph[-1, :] = 0
    dist_graph[0, -1] = 0
    np.fill_diagonal(dist_graph, 0)
    dist_graph[np.isinf(dist_graph)] = 0
    return dist_graph


def neg_log_p_dist2D(dist_px, diff_coeff, df):
    std = np.sqrt(2 * 2 * diff_coeff)
    p = 2 - 2 * stdtr(df, dist_px / (np.sqrt(2) * std))
    neg_log_p = np.ma.filled(-np.ma.log(p), np.inf)
    return neg_log_p


def calculate_dist_count(data):
    end_positions, start_positions = get_end_start_tracklets(data)
    dist_mat = cdist(
        np.zeros_like(end_positions["frame"].values).reshape([-1, 1]),
        np.float64(
            data.groupby("tracklet_id")["frame"].count().values.reshape([-1, 1])
        ),
    )
    dist_mat = np.pad(dist_mat, pad_width=1, constant_values=1)
    dist_mat[[0, -1], 1:-1] = dist_mat[1, 1:-1]
    dist_mat[1:-1, -1] = 1
    return dist_mat


def calculate_dist_confidence(data):
    end_positions, start_positions = get_end_start_tracklets(data)
    dist_mat = cdist(
        np.float64(np.zeros_like(end_positions["confidence"].values).reshape([-1, 1])),
        np.float64(
            data.groupby("tracklet_id")["confidence"].mean().values.reshape([-1, 1])
        ),
    )
    dist_mat = np.pad(dist_mat, pad_width=1, constant_values=1)
    dist_mat[[0, -1], 1:-1] = dist_mat[1, 1:-1]
    dist_mat[1:-1, -1] = 0.99
    return dist_mat


def calculate_dist_frame(data):
    end_positions, start_positions = get_end_start_tracklets(data)
    last_frame = np.max(data["frame"]).item()
    dist_mat = -np.subtract.outer(
        end_positions[["frame"]].values.T[0], start_positions[["frame"]].values.T[0]
    )
    dist_mat = np.pad(dist_mat, pad_width=1, constant_values=0)
    dist_mat[0, 1:-1] = start_positions[["frame"]].values.T[0] + 1
    dist_mat[1:-1, -1] = last_frame - end_positions[["frame"]].values.T[0] + 1
    dist_mat[-1, 0] = -last_frame
    dist_mat[-1, 1:-1] = start_positions[["frame"]].values.T[0] - last_frame
    return dist_mat


def calculate_dist_px(data):
    end_positions, start_positions = get_end_start_tracklets(data)
    dist_mat = cdist(
        np.float64(end_positions[["x", "y"]].values),
        np.float64(start_positions[["x", "y"]].values),
    )
    dist_mat = np.pad(dist_mat, pad_width=1, constant_values=1)
    return dist_mat


def get_end_start_tracklets(data):
    tracks = pd.DataFrame()
    tracks["start"] = (
        data.groupby("tracklet_id")["frame"].idxmin().values.astype("int").tolist()
    )
    tracks["end"] = (
        data.groupby("tracklet_id")["frame"].idxmax().values.astype("int").tolist()
    )
    start_positions = data.loc[tracks["start"]].reset_index()
    end_positions = data.loc[tracks["end"]].reset_index()
    return (end_positions, start_positions)


def save_longtracks_movie(
    video_path: str, out_dir: str, max_time: int, max_history: int, overwrite: bool
):

    video_name = os.path.basename(video_path).split(".")[0]

    TRACKING_VIDEO_FILE = f"{out_dir}/{video_name}_longtracks.mp4"
    TRACKING_FILE = f"{out_dir}/{video_name}_longtracks.json.gz"

    if os.path.exists(TRACKING_VIDEO_FILE) and (not overwrite):
        return

    with gzip.open(TRACKING_FILE, "rt") as f:
        all_tracked_objects = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * max_time]).astype(
        int
    )

    max_history_len = int(max_history * fps)
    tracking_history = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tracking_movie = cv2.VideoWriter(
        TRACKING_VIDEO_FILE,
        fourcc,
        fps,
        (int(width / 2), int(height / 2)),
    )

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_tracking = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        tracked_objects = all_tracked_objects[i]

        for track in tracked_objects:
            track_id = int(track[8])
            current_pos = (int(track[0]), int(track[1]))

            if track_id not in tracking_history:
                tracking_history[track_id] = deque(maxlen=max_history_len)

            tracking_history[track_id].append(current_pos)

            coords = (
                current_pos,
                (track[2] / 2, track[3] / 2),
                degrees(track[4]),
            )
            color = cv2.applyColorMap(
                np.array(((track_id * 29) % 255), dtype="uint8"), cv2.COLORMAP_HSV
            )[0][0]
            color = tuple(color.tolist())
            frame_tracking = cv2.ellipse(
                frame_tracking,
                coords,
                color,
                2,
            )
            frame_tracking = cv2.putText(
                frame_tracking,
                str(int(track_id)),
                tuple(map(int, np.add(coords[0], 15))),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                color=color,
                thickness=1,
            )
            tail_pts = np.array(list(tracking_history[track_id]), dtype=np.int32)
            frame_tracking = cv2.polylines(
                frame_tracking,
                [tail_pts],
                isClosed=False,
                color=color,
                thickness=2,
            )

        frame_tracking = cv2.resize(
            frame_tracking,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_LINEAR,
        )
        frame_tracking = cv2.putText(
            frame_tracking,
            f"{i:05}" + "/" + f"{total_frames:05}",
            (10, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color=(0, 0, 0),
            thickness=1,
        )
        tracking_movie.write(frame_tracking)

    tracking_movie.release()
    cap.release()


def load_results_to_df(
    video_path: str,
    out_dir: str,
    data_type: str,
):

    video_name = os.path.basename(video_path).split(".")[0]

    json_file = f"{out_dir}/{video_name}_{data_type}.json.gz"
    with gzip.open(json_file, "rt") as f:
        data_list = json.load(f)

    match data_type:
        case "detection" | "leftovers":
            columns = [
                "x",
                "y",
                "axis_1",
                "axis_2",
                "orientation",
                "confidence",
            ]
        case "tracking":
            columns = [
                "x",
                "y",
                "axis_1",
                "axis_2",
                "orientation",
                "tracklet_id",
                "confidence",
            ]
        case "longtracks":
            columns = [
                "x",
                "y",
                "axis_1",
                "axis_2",
                "orientation",
                "tracklet_id",
                "confidence",
                "frame",
                "track_id",
            ]
        case _:
            print("No matching file found.")

    list_df = []
    for i, frame in enumerate(data_list):
        objects = pd.DataFrame(
            frame,
            columns=columns,
        )
        objects["frame"] = i
        list_df.append(objects)

    results = pd.concat(list_df, ignore_index=True)
    results["frame"] = results["frame"].astype("int")
    if "tracklet_id" in results.columns:
        results["tracklet_id"] = results["tracklet_id"].astype("int")
        results["temp_id"] = results["tracklet_id"]
    if "track_id" in results.columns:
        results["track_id"] = results["track_id"].astype("int")
        results["temp_id"] = results["track_id"]

    results.sort_values(["temp_id", "frame"], inplace=True)
    results["diff_frame"] = -results.groupby("temp_id")["frame"].diff(periods=-1)
    results["diff_x"] = (
        -results.groupby("temp_id")["x"].diff(periods=-1) / results["diff_frame"]
    )
    results["diff_y"] = (
        -results.groupby("temp_id")["y"].diff(periods=-1) / results["diff_frame"]
    )
    results["diff_xy"] = np.sqrt(
        np.square(results["diff_x"]) + np.square(results["diff_y"])
    )
    results["diff_orientation"] = (
        -results.groupby("temp_id")["orientation"].diff(periods=-1)
        / results["diff_frame"]
    )
    results["diff_axis_1"] = (
        -results.groupby("temp_id")["axis_1"].diff(periods=-1) / results["diff_frame"]
    )
    results["diff_axis_2"] = (
        -results.groupby("temp_id")["axis_2"].diff(periods=-1) / results["diff_frame"]
    )

    results["cumul_dist"] = np.sqrt(
        np.square(results.groupby("temp_id")["x"].diff())
        + np.square(results.groupby("temp_id")["y"].diff())
    )
    results["cumul_dist"] = results.groupby("temp_id")["cumul_dist"].cumsum()

    results.drop("temp_id", inplace=True, axis=1)
    return results


def calculate_msd_lontracks(video_path: str, out_dir: str):

    TIME_RESOLUTION = 20
    MAX_REL_PERIOD = 0.5

    video_name = os.path.basename(video_path).split(".")[0]

    json_file = f"{out_dir}/{video_name}_longtracks.json.gz"
    with gzip.open(json_file, "rt") as f:
        data_list = json.load(f)

    columns = [
        "x",
        "y",
        "axis_1",
        "axis_2",
        "orientation",
        "tracklet_id",
        "confidence",
        "frame",
        "track_id",
    ]

    list_df = []
    for i, frame in enumerate(data_list):
        objects = pd.DataFrame(
            frame,
            columns=columns,
        )
        objects["frame"] = i
        list_df.append(objects)

    results = pd.concat(list_df, ignore_index=True)
    results["frame"] = results["frame"].astype("int")
    results["track_id"] = results["track_id"].astype("int")

    calculations = pd.DataFrame(
        product(
            results["track_id"].unique(),
            range(0, int(MAX_REL_PERIOD * max(results["frame"])), TIME_RESOLUTION),
        ),
        columns=["track_id", "period"],
    )
    calculations["msd"] = np.nan

    for id in results["track_id"].unique():
        data = results.loc[results["track_id"] == id]
        data.sort_values(["frame"], inplace=True)
        calculations.loc[
            (calculations["track_id"] == id) & (calculations["period"] == 0), "msd"
        ] = 0

        for p in range(
            TIME_RESOLUTION,
            min(
                calculations["period"].max(), data["frame"].max() - data["frame"].min()
            ),
            TIME_RESOLUTION,
        ):
            msd = data["x"].diff(periods=p).pow(2) + data["y"].diff(periods=p).pow(2)
            calculations.loc[
                (calculations["track_id"] == id) & (calculations["period"] == p), "msd"
            ] = msd.mean()

    return calculations


def calculate_vac_lontracks(video_path: str, out_dir: str):

    TIME_RESOLUTION = int(2)
    MAX_REL_PERIOD = 0.1

    video_name = os.path.basename(video_path).split(".")[0]

    json_file = f"{out_dir}/{video_name}_longtracks.json.gz"
    with gzip.open(json_file, "rt") as f:
        data_list = json.load(f)

    columns = [
        "x",
        "y",
        "axis_1",
        "axis_2",
        "orientation",
        "tracklet_id",
        "confidence",
        "frame",
        "track_id",
    ]

    list_df = []
    for i, frame in enumerate(data_list):
        objects = pd.DataFrame(
            frame,
            columns=columns,
        )
        objects["frame"] = i
        list_df.append(objects)

    results = pd.concat(list_df, ignore_index=True)
    results["frame"] = results["frame"].astype("int")
    results["track_id"] = results["track_id"].astype("int")

    calculations = pd.DataFrame(
        product(
            results["track_id"].unique(),
            range(
                0,
                int(
                    np.power(
                        MAX_REL_PERIOD * results["frame"].max(), 1 / TIME_RESOLUTION
                    )
                ),
            ),
        ),
        columns=["track_id", "period"],
    )
    calculations["period"] = calculations["period"].pow(TIME_RESOLUTION)
    calculations["vac"] = np.nan

    for id in results["track_id"].unique():
        data = results.loc[results["track_id"] == id]
        data.sort_values(["frame"], inplace=True)
        calculations.loc[
            (calculations["track_id"] == id) & (calculations["period"] == 0), "vac"
        ] = 1

        dx = data["x"].diff()
        dy = data["y"].diff()
        velocity = np.array([dx, dy]).T

        speed_squared = (dx.pow(2) + dy.pow(2)) / data["frame"].diff().pow(2)
        speed_squared = speed_squared.mean()

        periods = [
            p
            for p in calculations["period"].unique()
            if (p <= (max(data["frame"]) - min(data["frame"])) and (p > 0))
        ]

        for p in periods:
            vac = np.sum(velocity[p:, :] * velocity[:-p, :], axis=1)
            if np.any(~np.isnan(vac)):
                calculations.loc[
                    (calculations["period"] == p) & (calculations["track_id"] == id),
                    "vac",
                ] = (
                    np.nanmean(vac) / speed_squared
                )

    return calculations


def plot_longtracks_summary(video_path: str, out_dir: str, overwrite: bool):

    video_name = os.path.basename(video_path).split(".")[0]

    TRACKING_PLOTS = f"{out_dir}/{video_name}_longtracks_traces.png"
    SUMMARY_PLOTS = f"{out_dir}/{video_name}_longtracks_summary.png"
    if (
        os.path.exists(TRACKING_PLOTS)
        and os.path.exists(SUMMARY_PLOTS)
        and (not overwrite)
    ):
        return

    data = load_results_to_df(video_path, out_dir, "longtracks")
    total_frames = data["frame"].max() - data["frame"].min() + 1

    MIN_RELATIVE_LENGTH = 0.25
    MOVING_AVG_WINDOW = min(6000, total_frames)

    length = (data.groupby("track_id")["frame"].count() / total_frames).sort_values(
        ascending=False
    )
    top_track_ids = length.loc[length > MIN_RELATIVE_LENGTH].index
    data = data.loc[np.isin(data["track_id"], top_track_ids)]
    data["track_id"] = data["track_id"].astype("category")

    p1 = (
        pn.ggplot(data)
        + pn.aes(x="x", y="y", color="track_id")
        + pn.geom_path(size=0.5)
        + pn.coord_equal()
        + pn.theme_void()
        + pn.theme(legend_position="none")
        + pn.facet_wrap("track_id")
    )

    p2 = (
        pn.ggplot()
        + pn.aes(y="track_id", x="frame")
        + pn.geom_point(
            size=0.1,
            data=data,
            mapping=pn.aes(color="track_id"),
        )
        + pn.geom_point(
            color="black",
            size=0.5,
            data=data.loc[data["diff_frame"] > 1],
        )
    )

    p3 = (
        pn.ggplot(data.loc[data["diff_frame"] > 1])
        + pn.aes(x="track_id", y="diff_frame", color="track_id")
        + pn.geom_sina()
    )

    p4 = (
        pn.ggplot(data)
        + pn.aes(x="frame", y="cumul_dist", color="track_id")
        + pn.geom_line(size=0.5)
    )

    p5 = (
        pn.ggplot(data)
        + pn.aes(x="frame", y="diff_xy", color="track_id")
        + pn.geom_smooth(
            method="mavg",
            method_args={"window": MOVING_AVG_WINDOW},
            na_rm=True,
            se=False,
            size=0.5,
        )
    )

    p6 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="diff_xy", fill="track_id")
        + pn.geom_violin(width=1.5, size=0.1)
        + pn.scale_y_sqrt()
    )

    p7 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="orientation", fill="track_id")
        + pn.geom_violin(width=1, size=0.1)
    )

    p8 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="axis_1", fill="track_id")
        + pn.geom_violin(width=1, size=0.1)
    )

    p9 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="axis_2", fill="track_id")
        + pn.geom_violin(width=1, size=0.1)
    )

    data_msd = calculate_msd_lontracks(video_path, out_dir)
    data_msd = data_msd.loc[np.isin(data_msd["track_id"], top_track_ids)]
    data_msd["track_id"] = data_msd["track_id"].astype("category")

    p10 = (
        pn.ggplot(data_msd)
        + pn.aes(x="period", y="msd", color="track_id")
        + pn.geom_line(size=0.5)
    )

    data_vac = calculate_vac_lontracks(video_path, out_dir)
    data_vac = data_vac.loc[np.isin(data_vac["track_id"], top_track_ids)]
    data_vac["track_id"] = data_vac["track_id"].astype("category")

    p11 = (
        pn.ggplot(data_vac)
        + pn.aes(x="period", y="vac", color="track_id")
        + pn.geom_line(size=0.5)
        + pn.scale_x_sqrt()
    )

    plot = (
        (p2 | p3) / (p4 | p5) / (p6 | p7) / (p8 | p9) / (p10 | p11)
        & pn.theme_minimal()
        & pn.theme(figure_size=(8.268, 11.693), legend_position="none")
    )

    p1.save(TRACKING_PLOTS, width=8.268, height=11.693, dpi=600)
    plot.save(SUMMARY_PLOTS, dpi=600)
