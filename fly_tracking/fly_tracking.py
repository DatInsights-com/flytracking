# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: May 7, 2026
# Version: 1.1

import os

import cv2
import numpy as np

from math import degrees

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from collections import deque

from ultralytics.trackers.byte_tracker import BYTETracker

import json
import gzip

TRACKER_TYPE = "bytetrack"
TRACK_HIGH_THRESH = 0
TRACK_LOW_THRESH = 0
NEW_TRACK_THRESH = 0.1
TRACK_BUFFER = 0
MATCH_THRESH = 0.9999
FUSE_SCORE = True

N_TRACK_COLORS = 16


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


def track_moving_objects(
    video_path: str, out_dir: str, total_frames: int, overwrite: bool
):

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

    with gzip.open(DETECTION_FILE, "rt") as f:
        all_detected_objects = json.load(f)

    cfg = Tracker_cfg(
        TRACKER_TYPE,
        TRACK_HIGH_THRESH,
        TRACK_LOW_THRESH,
        NEW_TRACK_THRESH,
        TRACK_BUFFER,
        MATCH_THRESH,
        FUSE_SCORE,
    )
    tracker = BYTETracker(cfg)

    all_tracked_objects = [[] for _ in range(total_frames)]
    not_tracked_objects = [[] for _ in range(total_frames)]

    for i in range(total_frames):

        if not all_tracked_objects[i]:
            objects_ellipses = all_detected_objects[i]
            xywhr = []
            conf = []
            obj_cls = []
            for obj in objects_ellipses:
                xywhr.append(obj[:5])
                conf.append(obj[5])
                obj_cls.append(0)

            objects = Detections(np.array(xywhr), np.array(conf), np.array(obj_cls))

            tracked_objects = tracker.update(objects)

            if tracked_objects.ndim == 2:
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
            else:
                not_tracked_objects[i] = all_detected_objects[i]

    with gzip.open(TRACKING_FILE, "wt") as f:
        json.dump(all_tracked_objects, f)

    with gzip.open(LEFTOVER_FILE, "wt") as f:
        json.dump(not_tracked_objects, f)


def save_tracking_movie(
    video_path: str, out_dir: str, total_frames: int, max_history: int, overwrite: bool
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
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), total_frames]).astype(int)

    max_history_len = int(max_history * fps)
    tracking_history = {}

    colors_bgr = []
    for h in np.linspace(0, 180, N_TRACK_COLORS, endpoint=False):
        bgr = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors_bgr.append(tuple(bgr.tolist()))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    tracking_movie = cv2.VideoWriter(
        TRACKING_VIDEO_FILE,
        fourcc,
        fps,
        (int(width / 2), int(height / 2)),
    )

    for i in range(total_frames):
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
            track_id = int(track[5])
            current_pos = (int(track[0]), int(track[1]))

            if track_id not in tracking_history:
                tracking_history[track_id] = deque(maxlen=max_history_len)

            tracking_history[track_id].append(current_pos)

            coords = (
                (track[0], track[1]),
                (track[2] / 2, track[3] / 2),
                degrees(track[4]),
            )
            color = colors_bgr[track_id % 16]
            frame_tracking = cv2.ellipse(
                frame_tracking,
                coords,
                color,
                2,
            )
            tail_pts = np.array(list(tracking_history[track_id]), dtype=np.int32)
            frame_tracking = cv2.polylines(
                frame_tracking,
                [tail_pts],
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
            interpolation=cv2.INTER_AREA,
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
