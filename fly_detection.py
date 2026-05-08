# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: May 7, 2026
# Version: 1.1

import numpy as np
import cv2

import threading
import os

from multiprocessing import Process, JoinableQueue, Queue
from multiprocessing import shared_memory

from scipy.stats import chi2
from math import radians, degrees

import json
import gzip

N_MAX_OBJS_PER_FRAME = 100
N_WORKERS = 16
N_FRAME_QUEUE = 400

MIN_INTESITY = 7
MIN_AREA = 7
GAUSSIAN_STD = 11
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5, 5])


def create_background_image(
    video_path: str, out_dir: str, nb_frames: int, total_frames: int, overwrite: bool
):
    video_name = os.path.basename(video_path).split(".")[0]

    BACKGROUND_FILE = f"{out_dir}/{video_name}_background.png"

    if os.path.exists(BACKGROUND_FILE) and (not overwrite):
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    stride = int(np.max([1, total_frames / nb_frames]))
    background = np.zeros((height, width), dtype=np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(nb_frames):
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


def detect_moving_objects(
    video_path: str, out_dir: str, total_frames: int, overwrite: bool
):

    video_name = os.path.basename(video_path).split(".")[0]

    DETECTION_FILE = f"{out_dir}/{video_name}_detection.json.gz"
    BACKGROUND_FILE = f"{out_dir}/{video_name}_background.png"

    if os.path.exists(DETECTION_FILE) and (not overwrite):
        return

    bgframe = cv2.imread(BACKGROUND_FILE, cv2.IMREAD_GRAYSCALE)
    frame_width = int(bgframe.shape[1])
    frame_height = int(bgframe.shape[0])

    frames_shared_memory = shared_memory.SharedMemory(
        create=True, size=N_FRAME_QUEUE * frame_width * frame_height * 4
    )

    results_shared_memory = shared_memory.SharedMemory(
        create=True,
        size=total_frames * N_MAX_OBJS_PER_FRAME * 6 * 4,
    )
    results_shared_memory.buf[: results_shared_memory.size] = (
        b"\x00" * results_shared_memory.size
    )

    frame_q = JoinableQueue(maxsize=N_FRAME_QUEUE)

    cap_thread = threading.Thread(
        target=capture_worker,
        args=(
            video_path,
            total_frames,
            bgframe,
            frames_shared_memory.name,
            frame_q,
        ),
        daemon=True,
    )
    cap_thread.start()

    workers = []
    for id in range(N_WORKERS):
        p = Process(
            target=worker_process,
            args=(
                frame_q,
                frames_shared_memory.name,
                results_shared_memory.name,
                frame_width,
                frame_height,
                total_frames,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    frame_q.join()

    read_results = shared_memory.SharedMemory(name=results_shared_memory.name)
    results_array = np.ndarray(
        (total_frames, N_MAX_OBJS_PER_FRAME, 6),
        dtype=np.float32,
        buffer=read_results.buf,
    )

    all_detected_objects = [[] for _ in range(total_frames)]

    for i in range(results_array.shape[0]):
        for j in range(results_array.shape[1]):
            if results_array[i, j, 5]:
                all_detected_objects[i].append(results_array[i, j, :].tolist())

    with gzip.open(DETECTION_FILE, "wt") as f:
        json.dump(all_detected_objects, f)

    frames_shared_memory.close()
    frames_shared_memory.unlink()

    results_shared_memory.close()
    results_shared_memory.unlink()


def capture_worker(
    video_path, total_frames, bgframe, frames_shared_memory_name, frame_q
):

    frames_shared_memory = shared_memory.SharedMemory(name=frames_shared_memory_name)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), total_frames]).astype(int)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_nb in range(total_frames):
        slot = frame_nb % N_FRAME_QUEUE
        frame_buffer = np.ndarray(
            (height, width),
            dtype=np.float32,
            buffer=frames_shared_memory.buf,
            offset=slot * width * height * 4,
        )

        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_frame = cv2.subtract(
            bgframe.astype(np.float32), frame_gray.astype(np.float32)
        )

        np.copyto(frame_buffer, fg_frame)

        frame_q.put(frame_nb)
    cap.release()

    for _ in range(N_WORKERS):
        frame_q.put(None)

    frames_shared_memory.close()


def worker_process(
    frame_q,
    frames_shared_memory_name,
    results_shared_memory_name,
    frame_width,
    frame_height,
    total_frames,
):

    frames_shared_memory = shared_memory.SharedMemory(name=frames_shared_memory_name)
    results_shared_memory = shared_memory.SharedMemory(name=results_shared_memory_name)

    results_array = np.ndarray(
        (total_frames, N_MAX_OBJS_PER_FRAME, 6),
        dtype=np.float32,
        buffer=results_shared_memory.buf,
    )

    while True:
        frame_nb = frame_q.get()
        if frame_nb is None:
            frame_q.task_done()
            break

        # Get frame data
        slot = frame_nb % N_FRAME_QUEUE
        fg_frame = np.ndarray(
            (frame_height, frame_width),
            dtype=np.float32,
            buffer=frames_shared_memory.buf,
            offset=slot * frame_height * frame_width * 4,
        )

        objects_rectangles = process_frame_detection(fg_frame)

        count = min(len(objects_rectangles), N_MAX_OBJS_PER_FRAME)

        # objects_rectangles is [x, y, rx, ry, angle, confidence]
        for i in range(count):
            results_array[frame_nb, i, :] = np.array(
                objects_rectangles[i], dtype=np.float32
            )

        frame_q.task_done()

    frames_shared_memory.close()
    results_shared_memory.close()


def process_frame_detection(fg_frame):

    fg_smooth = cv2.subtract(
        cv2.GaussianBlur(fg_frame, (GAUSSIAN_STD, GAUSSIAN_STD), 0),
        cv2.GaussianBlur(fg_frame, (GAUSSIAN_STD * 3, GAUSSIAN_STD * 3), 0),
    )

    fg_mask = cv2.threshold(fg_smooth, MIN_INTESITY, 255, cv2.THRESH_BINARY)[1]
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, MORPH_KERNEL).astype(np.uint8)

    total_labels, label_ids = cv2.connectedComponents(fg_mask, 4, cv2.CV_32S)

    for id in range(1, total_labels):
        mask = label_ids == id
        max_int = np.max(fg_smooth[mask])
        label_ids[mask & (fg_smooth < np.max([MIN_INTESITY, (max_int / 3)]))] = 0

    fg_mask = (255 * (label_ids > 0)).astype(np.uint8)
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
            if (len(cnt) > 4) and (ar > MIN_AREA)
        ]
    )

    objects_rectangles = [cv2.fitEllipse(cnt) for cnt in contours]
    objects_rectangles = [
        obj
        for obj in objects_rectangles
        if (obj[0][0] < fg_mask.shape[1])
        and (obj[0][1] < fg_mask.shape[0])
        and (obj[0][0] >= 0)
        and (obj[0][1] >= 0)
    ]
    # [y, x, axis_long/2, axis_short/2, angle_degree]
    objects_intensities = [
        fg_smooth[int(x), int(y)] for (y, x), (_, _), _ in objects_rectangles
    ]
    objects_properties = np.stack(
        [
            [o[1][0] for o in objects_rectangles],
            [o[1][1] for o in objects_rectangles],
            objects_intensities,
        ],
        axis=1,
    )
    confidence = calculate_confidence(objects_properties)

    objects_rectangles = [
        [a, b, 2.5 * c, 2.5 * d, radians(e), f]
        for ((a, b), (c, d), e), (f) in zip(objects_rectangles, confidence)
        if f > 0
    ]

    return objects_rectangles


def calculate_confidence(properties):
    ind = np.any(properties <= 0, axis=1)
    properties[ind, :] = 1
    log_prop = np.log(properties)
    distances = np_mahalanobis_distance2(log_prop)
    confidence = 1 - chi2.cdf(distances, 3)
    confidence[ind] = 0
    return confidence


def np_mahalanobis_distance2(properties):
    mu = np.mean(properties, axis=0, keepdims=False)
    M = properties - mu
    cov = 1.0 / (properties.shape[0] - 1) * np.dot(M.T, M)
    X_mu_SInv = np.dot(M, np.linalg.inv(cov))
    return np.sum(X_mu_SInv * M, axis=1)


def save_detection_movie(
    video_path: str, out_dir: str, total_frame: int, overwrite: bool
):

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
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), total_frame]).astype(int)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    detection_movie = cv2.VideoWriter(
        DETECTION_VIDEO_FILE,
        fourcc,
        fps,
        (int(width / 2), int(height / 2)),
    )

    for i in range(total_frames):
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
