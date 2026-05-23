# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: May 7, 2026
# Version: 1.1

import os

import cv2
import numpy as np
import pandas as pd
from math import degrees

from scipy.spatial.distance import cdist
from scipy.special import stdtr
from scipy.sparse.csgraph import shortest_path

from collections import deque

import json
import gzip

PROB_NO_DETECTION = 1e-3
PROB_DOUBLE_DETECTION = 1e-4
T_DIST_DF = 11
MIN_PROB_THRESH = 0
MIN_OBJECT_CONFIDENCE = 0.25

N_TRACK_COLORS = 16


def connect_tracklets(
    video_path: str, out_dir: str, total_frames: int, overwrite: bool
):

    video_name = os.path.basename(video_path).split(".")[0]

    LONG_TRACKS_FILE = f"{out_dir}/{video_name}_longtracks.json.gz"
    TRACKING_FILE = f"{out_dir}/{video_name}_tracking.json.gz"
    LEFTOVER_FILE = f"{out_dir}/{video_name}_leftovers.json.gz"

    if os.path.exists(LONG_TRACKS_FILE) and (not overwrite):
        return

    with gzip.open(TRACKING_FILE, "rt") as f:
        all_tracked_objects = json.load(f)

    with gzip.open(LEFTOVER_FILE, "rt") as f:
        leftover_objects = json.load(f)

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

    all_tracks = []
    tracklet_id = objects_df["tracklet_id"].unique()
    for id in tracklet_id:
        tracks = objects_df.loc[objects_df["tracklet_id"] == id]
        tracks.index = tracks["frame"]
        if len(tracks) > 1:
            tracks = tracks.reindex(
                np.arange(tracks["frame"].min(), tracks["frame"].max() + 1),
                fill_value=np.nan,
            ).interpolate()
        all_tracks.append(tracks)
    interpolated_df = pd.concat(all_tracks, ignore_index=True)
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

    data_relabeled = data_relabeled.sort_values(by=["frame", "tracklet_id"])
    data_relabeled = data_relabeled.drop_duplicates(
        ["frame", "long_track_id"], keep="first", ignore_index=True
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
    video_path: str, out_dir: str, total_frames: int, max_history: int, overwrite: bool
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
            color = colors_bgr[track_id % 16]
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
