# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: May 7, 2026
# Version: 1.1

import gzip
import json
import os
import threading
from math import degrees, radians
from multiprocessing import JoinableQueue, Process, shared_memory

import cv2
import numpy as np
from scipy.stats import chi2

N_MAX_OBJS_PER_FRAME = 50
N_WORKERS = 4
N_FRAME_QUEUE = N_WORKERS * 100

MIN_INTENSITY = 5
REL_INTENSITY = 0.3
MIN_AREA = 2
GAUSSIAN_STD = 3
DOG_FACTOR = 1.6  # scaling factor difference of Gaussian
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5, 5])

TAG_1_FILE = "flytracking/patterns/tag_1.png"
TAG_2_FILE = "flytracking/patterns/tag_2.png"
HOLE_PATTERN_FILE = "flytracking/patterns/hole_pattern.png"
HOLES_MASK_FILE = "flytracking/patterns/holes_mask.png"
HOLE_LABELS = [6, 1, 2, 3, 4, 5]
MORPH_KERNEL_HOLES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [20, 20])


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def register_background_coordinates(video_path: str, out_dir: str, overwrite: bool):
    video_name = os.path.basename(video_path).split(".")[0]

    BACKGROUND_FILE = f"{out_dir}/{video_name}_background_registered.png"
    COORDINATES_FILE = f"{out_dir}/{video_name}_coordinates.json.gz"

    if (
        os.path.exists(BACKGROUND_FILE)
        and os.path.exists(COORDINATES_FILE)
        and (not overwrite)
    ):
        return

    coordinates = {}

    bg_path = f"{out_dir}/{video_name}_background.png"
    tag_1 = cv2.imread(TAG_1_FILE, cv2.IMREAD_GRAYSCALE)
    tag_2 = cv2.imread(TAG_2_FILE, cv2.IMREAD_GRAYSCALE)
    holes_mask = cv2.imread(HOLES_MASK_FILE, cv2.IMREAD_GRAYSCALE)
    hole_pattern = cv2.imread(HOLE_PATTERN_FILE, cv2.IMREAD_GRAYSCALE)
    hole_pattern_offset = np.array(hole_pattern.shape) / 2

    bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    bg_w, bg_h = bg_img.shape
    bg_img_corner = bg_img[int(bg_w * 0.75) :, int(bg_h * 0.75) :]

    holes_mask = holes_mask[
        int(hole_pattern_offset[0] + 1024 - bg_w / 2) : int(
            bg_w - hole_pattern_offset[0] + 1024 - bg_w / 2
        )
        + 1,
        int(hole_pattern_offset[1] + 1024 - bg_h / 2) : int(
            bg_w - hole_pattern_offset[1] + 1024 - bg_h / 2
        )
        + 1,
    ]

    res_1 = np.max(cv2.matchTemplate(bg_img_corner, tag_1, cv2.TM_CCOEFF_NORMED))
    res_2 = np.max(cv2.matchTemplate(bg_img_corner, tag_2, cv2.TM_CCOEFF_NORMED))

    if res_1 > res_2:
        plate = 1
    else:
        plate = 2

    coordinates.update({"plate_id": plate})

    holes = cv2.matchTemplate(bg_img, hole_pattern, cv2.TM_CCOEFF_NORMED)
    holes[holes < 0.2] = 0
    holes = holes * holes_mask
    holes = holes == cv2.dilate(holes, MORPH_KERNEL_HOLES, iterations=20)
    holes = (holes * holes_mask).astype(np.uint8)
    holes = cv2.dilate(holes, MORPH_KERNEL, iterations=1)
    contours, _ = cv2.findContours(
        holes,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    objects_circles = [cv2.minEnclosingCircle(cnt) for cnt in contours]
    objects_intensities = np.array(
        [holes[round(b), round(a)] for (a, b), _ in objects_circles]
    )
    rank_intensities = np.argsort(-objects_intensities)[:6].tolist()

    objects_circles = [
        ((a + hole_pattern_offset[0], b + hole_pattern_offset[1]), r)
        for i, ((a, b), r) in enumerate(objects_circles)
        if i in rank_intensities
    ]

    holes_coords = np.expand_dims(
        np.array(
            [(a, b) for ((a, b), r) in objects_circles],
            dtype=np.float32,
        ),
        1,
    )
    e = cv2.fitEllipse(holes_coords)
    coordinates.update({"ellipse_holes": e})

    pol_coords = [
        cart2pol(
            a - e[0][0],
            b - e[0][1],
        )
        for ((a, b), r) in objects_circles
    ]
    ind_phi = np.argsort(np.array(pol_coords)[:, 1]).argsort()

    bg_img_overlay = bg_img
    cv2.putText(
        bg_img_overlay,
        "plate: " + str(plate),
        [1600, 100],
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        2,
    )
    cv2.ellipse(bg_img_overlay, e, (0, 200, 0), 2)
    cv2.circle(bg_img_overlay, np.array(e[0], dtype=np.int32), 10, (0, 200, 0), 2)

    for i, c in enumerate(objects_circles):
        coordinates.update({"hole_" + str(HOLE_LABELS[ind_phi[i]]): c[0]})
        cv2.circle(
            bg_img_overlay,
            np.array(c[0], dtype=np.int32),
            20,
            [50, 50, 50],
            -1,
        )
        cv2.putText(
            bg_img_overlay,
            str(HOLE_LABELS[ind_phi[i]]),
            np.array(c[0], dtype=np.int32) + [-15, -35],
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2,
        )

    cv2.imwrite(BACKGROUND_FILE, bg_img_overlay)
    with gzip.open(COORDINATES_FILE, "wt") as f:
        json.dump(coordinates, f)


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
    total_frames = np.min([cap.get(cv2.CAP_PROP_FRAME_COUNT), total_frames]).astype(int)
    nb_frames = np.min([nb_frames, total_frames])

    stride = int(np.max([1, total_frames / nb_frames]))
    background = np.zeros((height, width), dtype=np.float32)

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

    frame_q = JoinableQueue(maxsize=N_FRAME_QUEUE - N_WORKERS)

    workers = []
    for _ in range(N_WORKERS):
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
        workers.append(p)
        p.start()

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
    cap_thread.join()
    frame_q.join()

    for _ in range(N_WORKERS):
        frame_q.put(None)

    for p in workers:
        p.join()

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
        frame_q.put(frame_nb)
        np.copyto(frame_buffer, fg_frame)
    cap.release()
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
        try:
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
            # objects_rectangles is [x, y, rx, ry, angle, confidence]
            if len(objects_rectangles) < N_MAX_OBJS_PER_FRAME:
                for i in range(len(objects_rectangles)):
                    results_array[frame_nb, i, :] = np.array(
                        objects_rectangles[i], dtype=np.float32
                    )
        finally:
            if frame_nb is not None and frame_nb is not False:
                frame_q.task_done()
    frames_shared_memory.close()
    results_shared_memory.close()


def process_frame_detection(fg_frame):

    fg_smooth = cv2.subtract(
        cv2.GaussianBlur(fg_frame, (0, 0), GAUSSIAN_STD),
        cv2.GaussianBlur(fg_frame, (0, 0), GAUSSIAN_STD * DOG_FACTOR),
    )

    fg_mask = cv2.threshold(fg_smooth, MIN_INTENSITY, 255, cv2.THRESH_BINARY)[1].astype(
        np.uint8
    )
    fg_dilate_maxima = cv2.dilate(fg_smooth, MORPH_KERNEL)

    fg_mask = (fg_mask > 0) & (fg_smooth > (REL_INTENSITY * fg_dilate_maxima)).astype(
        np.uint8
    )
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    fg_maxima_mask = (fg_smooth == fg_dilate_maxima) & fg_mask
    fg_maxima = np.transpose(np.where(fg_maxima_mask)).astype(np.float32)
    fg_maxima[:, [0, 1]] = fg_maxima[:, [1, 0]]

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    all_contours = []
    for cnt in contours:
        n_max = 0
        for pt in fg_maxima:
            if cv2.pointPolygonTest(cnt, pt, False) > 0:
                n_max += 1
        if n_max > 1:
            supp_cnts = watershed_segmentation(cnt, fg_smooth, fg_mask, fg_maxima_mask)
            for c in supp_cnts:
                all_contours.append(c)
        else:
            all_contours.append(cnt)
  
    contours_area = [cv2.contourArea(cnt) for cnt in all_contours]

    if len(all_contours)>0:
        all_contours, contours_area = zip(
            *[
                (cnt, ar)
                for (cnt, ar) in zip(all_contours, contours_area)
                if (len(cnt) > 4) and (ar > MIN_AREA)
            ]
        )

        objects_rectangles = [cv2.minAreaRect(cnt) for cnt in all_contours]
        objects_rectangles = [
            obj
            for obj in objects_rectangles
            if (obj[0][0] < fg_mask.shape[1])
            and (obj[0][1] < fg_mask.shape[0])
            and (obj[0][0] >= 0)
            and (obj[0][1] >= 0)
        ]
        objects_rectangles = [
            ([(y, x), (ax1, ax2), r] if ax1 < ax2 else [(y, x), (ax2, ax1), r + 90])
            for (y, x), (ax1, ax2), r in objects_rectangles
        ]
        # [y, x, axis_1/2, axis_2/2, angle_degree]
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

        if objects_properties.shape[0]>2:
            confidence = calculate_confidence(objects_properties)
        else:
            confidence = np.ones(objects_properties.shape[0])

        objects_rectangles = [
            [a, b, 2.5 * c, 2.5 * d, radians(e), f]
            for ((a, b), (c, d), e), (f) in zip(objects_rectangles, confidence)
            if f > 0
        ]
    else:
        objects_rectangles = []

    return objects_rectangles


def watershed_segmentation(cnt, fg_smooth, fg_mask, fg_maxima_mask):
    x, y, w, h = cv2.boundingRect(cnt)
    rect_smooth = fg_smooth[y : y + h, x : x + w]
    rect_mask = fg_mask[y : y + h, x : x + w]
    rect_max = fg_maxima_mask[y : y + h, x : x + w]
    dist = cv2.normalize(
        cv2.distanceTransform(1 - rect_max, cv2.DIST_L2, 5),
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    ) * (np.max(rect_smooth) - rect_smooth)

    _, markers = cv2.connectedComponents(rect_max)
    markers[markers > 0] = markers[markers > 0] + 1
    markers[rect_mask == 0] = 1
    dist = cv2.cvtColor(
        cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    markers = cv2.watershed(dist, markers)
    contours = []
    for i in np.unique(markers[markers > 1]):
        cnt, _ = cv2.findContours(
            (markers == i).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours.append(cnt[0])
    contours = [c + (x, y) for c in contours]
    return contours


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
    if np.linalg.det(cov) != 0:
        X_mu_SInv = np.dot(M, np.linalg.inv(cov))
        distances = np.sum(X_mu_SInv * M, axis=1)
    else:
        distances = np.ones(M.shape[0])
    return distances


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

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
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
            interpolation=cv2.INTER_AREA,
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
