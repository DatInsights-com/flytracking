# Author: Yann Dufour
# Company: DatInsight, https://datinsights.com/
# Date: May 7, 2026
# Version: 1.1

import os

import json
import gzip
import pandas as pd
import numpy as np
import matplotlib
import plotnine as pn
from itertools import product

TIME_RESOLUTION_MSD = int(20)
MAX_REL_PERIOD_MSD = 0.25

TIME_RESOLUTION_VAC = int(2)
MAX_REL_PERIOD_VAC = 0.1

MIN_RELATIVE_LENGTH = 0.25
MOVING_AVG_WINDOW = int(1200)

PIXEL_SIZE = 0.1975  # mm/px
FRAME_DURATION = 1 / 20  # seconds
MAX_DIST_CENTER = 210  # mm

matplotlib.use("agg")


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
        objects = pd.DataFrame(frame, columns=columns, dtype=np.float64)
        objects["frame"] = i
        list_df.append(objects)

    results = pd.concat(list_df, ignore_index=True)
    results["frame"] = results["frame"].astype(np.int32)
    if "tracklet_id" in results.columns:
        results["tracklet_id"] = results["tracklet_id"].astype(np.int32)
        results["temp_id"] = results["tracklet_id"]
    if "track_id" in results.columns:
        results["track_id"] = results["track_id"].astype(np.int32)
        results["temp_id"] = results["track_id"]

    results = results.sort_values(["temp_id", "frame"])
    results["diff_frame"] = -results.groupby("temp_id")["frame"].diff(periods=-1)
    results["diff_x"] = (
        -results.groupby("temp_id")["x"].diff(periods=-1) / results["diff_frame"]
    )
    results["diff_y"] = (
        -results.groupby("temp_id")["y"].diff(periods=-1) / results["diff_frame"]
    )
    results["diff_xy"] = (results["diff_x"] ** 2 + results["diff_y"] ** 2) ** 0.5
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

    results["cumul_dist"] = (
        (
            (results.groupby("temp_id")["x"].diff()) ** 2
            + (results.groupby("temp_id")["y"].diff()) ** 2
        )
        ** 0.5
    ).astype(np.float64)
    results["cumul_dist"] = results.groupby("temp_id")["cumul_dist"].cumsum()

    results = results.drop("temp_id", axis=1)
    return results


def calculate_msd_lontracks(data):

    calculations = pd.DataFrame(
        product(
            data["track_id"].unique(),
            range(0, int(MAX_REL_PERIOD_MSD * max(data["frame"])), TIME_RESOLUTION_MSD),
        ),
        columns=["track_id", "period"],
    )
    calculations["msd"] = np.nan

    for id in data["track_id"].unique():
        track_data = data.loc[data["track_id"] == id]
        track_data = track_data.sort_values(["frame"])
        calculations.loc[
            (calculations["track_id"] == id) & (calculations["period"] == 0), "msd"
        ] = 0

        for p in range(
            TIME_RESOLUTION_MSD,
            min(
                calculations["period"].max(),
                track_data["frame"].max() - track_data["frame"].min(),
            ),
            TIME_RESOLUTION_MSD,
        ):
            msd = track_data["x"].diff(periods=p).pow(2) + track_data["y"].diff(
                periods=p
            ).pow(2)
            calculations.loc[
                (calculations["track_id"] == id) & (calculations["period"] == p), "msd"
            ] = msd.mean()

    calculations["track_id"] = calculations["track_id"].astype("category")
    return calculations


def calculate_vac_lontracks(data):

    calculations = pd.DataFrame(
        product(
            data["track_id"].unique(),
            range(
                0,
                int(
                    np.power(
                        MAX_REL_PERIOD_VAC * data["frame"].max(),
                        1 / TIME_RESOLUTION_VAC,
                    )
                ),
            ),
        ),
        columns=["track_id", "period"],
    )
    calculations["period"] = calculations["period"].pow(TIME_RESOLUTION_VAC)
    calculations["vac"] = np.nan

    for id in data["track_id"].unique():
        track_data = data.loc[data["track_id"] == id]
        track_data = track_data.sort_values(["frame"])
        calculations.loc[
            (calculations["track_id"] == id) & (calculations["period"] == 0), "vac"
        ] = 1

        dx = track_data["x"].diff().to_numpy(dtype="float64")
        dy = track_data["y"].diff().to_numpy(dtype="float64")
        velocity = np.array([dx, dy]).T

        speed_squared = (np.square(dx) + np.square(dy)) / np.square(
            track_data["frame"].diff().to_numpy()
        )
        speed_squared = np.nanmean(speed_squared)

        periods = [
            p
            for p in calculations["period"].unique()
            if (p <= (max(track_data["frame"]) - min(track_data["frame"])) and (p > 0))
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

    calculations["track_id"] = calculations["track_id"].astype("category")
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

    length = (data.groupby("track_id")["frame"].count() / total_frames).sort_values(
        ascending=False
    )
    top_track_ids = length.loc[length > MIN_RELATIVE_LENGTH].index
    data = data.loc[np.isin(data["track_id"], top_track_ids)]
    data["track_id"] = data["track_id"].astype("category")

    p1 = (
        pn.ggplot(data)
        + pn.aes(x="x", y="y", color="track_id")
        + pn.geom_path(size=0.5, na_rm=True)
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
            na_rm=True,
        )
        + pn.geom_point(
            color="black",
            size=0.5,
            data=data.loc[data["diff_frame"] > 1], 
            na_rm=True,
        )
    )
    if (data.loc[data["diff_frame"] > 1]).empty:
        p3 = pn.ggplot() + pn.geom_blank()
    else:
        p3 = (
            pn.ggplot(data.loc[data["diff_frame"] > 1])
            + pn.aes(x="track_id", y="diff_frame", color="track_id")
            + pn.geom_jitter(alpha = 0.5, na_rm=True)
        )

    p4 = (
        pn.ggplot(data)
        + pn.aes(x="frame", y="cumul_dist", color="track_id")
        + pn.geom_line(size=0.5, na_rm=True)
    )

    p5 = (
        pn.ggplot(data)
        + pn.aes(x="frame", y="diff_xy", color="track_id")
        + pn.geom_smooth(
            method="mavg",
            method_args={"window": min(MOVING_AVG_WINDOW, total_frames)},
            na_rm=True,
            se=False,
            size=0.5,
        )
    )

    p6 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="diff_xy", fill="track_id")
        + pn.geom_violin(width=1, size=0.1, na_rm=True)
        + pn.scale_y_continuous(limits=[0, 25], trans="sqrt")
    )

    p7 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="orientation", fill="track_id")
        + pn.geom_violin(width=1, size=0.1, na_rm=True)
    )

    p8 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="axis_1", fill="track_id")
        + pn.geom_violin(width=1, size=0.1, na_rm=True)
    )

    p9 = (
        pn.ggplot(data)
        + pn.aes(x="track_id", y="axis_2", fill="track_id")
        + pn.geom_violin(width=1, size=0.1, na_rm=True)
    )
    data_msd = calculate_msd_lontracks(data)

    p10 = (
        pn.ggplot(data_msd)
        + pn.aes(x="period", y="msd", color="track_id")
        + pn.geom_line(size=0.5,na_rm=True)
    )
    data_vac = calculate_vac_lontracks(data)

    p11 = (
        pn.ggplot(data_vac)
        + pn.aes(x="period", y="vac", color="track_id")
        + pn.geom_line(size=0.5, na_rm=True)
        + pn.scale_x_sqrt()
    )

    plot = (
        (p2 | p3) / (p4 | p5) / (p6 | p7) / (p8 | p9) / (p10 | p11)
        & pn.theme_minimal()
        & pn.theme(figure_size=(8.268, 11.693), legend_position="none")
    )

    p1.save(TRACKING_PLOTS, width=8.268, height=11.693, dpi=600, verbose=False)
    plot.save(SUMMARY_PLOTS, dpi=600, verbose=False)


def tracks_to_dataframe(video_path: str, out_dir: str, overwrite: bool):
    video_name = os.path.basename(video_path).split(".")[0]

    DATAFRAME_FILE = f"{out_dir}/{video_name}_dataframe.json.gz"
    if os.path.exists(DATAFRAME_FILE) and (not overwrite):
        return

    json_file = f"{out_dir}/{video_name}_longtracks.json.gz"
    coordinates_file = f"{out_dir}/{video_name}_coordinates.json.gz"

    with gzip.open(json_file, "rt") as f:
        data_list = json.load(f)

    with gzip.open(coordinates_file, "rt") as f:
        coordinates = json.load(f)

    parts = out_dir.split("/")
    exp_name = parts[-2]
    group_name = parts[-1]
    rep_name = (
        video_name.replace("2_butanone", "2-butanone")
        .replace("mineral_oil", "mineral-oil")
        .split("_")
    )

    coord_center = coordinates["ellipse_holes"][0]
    coord_odor = (
        np.array(coordinates["hole_" + rep_name[3]]) - np.array(coord_center)
    ) * PIXEL_SIZE

    cos_rot = coord_odor[0] / np.linalg.norm(coord_odor)
    sin_rot = -coord_odor[1] / np.linalg.norm(coord_odor)
    odor_rot = np.atan2(-coord_odor[1], coord_odor[0])

    coord_odor = np.array([np.linalg.norm(coord_odor), 0])

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
        objects = pd.DataFrame(frame, columns=columns, dtype=np.float64)
        objects = objects.drop("tracklet_id", axis=1)
        objects["frame"] = i
        list_df.append(objects)

    tracks = pd.concat(list_df, ignore_index=True)
    tracks["x"] = (tracks["x"] - coord_center[0]) * PIXEL_SIZE
    tracks["y"] = (tracks["y"] - coord_center[1]) * PIXEL_SIZE

    tracks["x_rot"] = (tracks["x"] * cos_rot) - (tracks["y"] * sin_rot)
    tracks["y"] = (tracks["x"] * sin_rot) + (tracks["y"] * cos_rot)
    tracks["x"] = tracks["x_rot"]
    tracks.drop("x_rot", axis=1)
    tracks["orientation"] = tracks["orientation"] + odor_rot

    tracks["axis_1"] = tracks["axis_1"] * PIXEL_SIZE
    tracks["axis_2"] = tracks["axis_2"] * PIXEL_SIZE
    tracks["frame"] = tracks["frame"].astype(np.int32)
    tracks["track_id"] = tracks["track_id"].astype(np.int32)
    tracks["time"] = tracks["frame"] * FRAME_DURATION

    tracks = tracks.sort_values(["track_id", "frame"])
    tracks["diff_time"] = -tracks.groupby("track_id")["time"].diff(periods=-1)
    tracks["diff_x"] = (
        -tracks.groupby("track_id")["x"].diff(periods=-1) / tracks["diff_time"]
    )
    tracks["diff_y"] = (
        -tracks.groupby("track_id")["y"].diff(periods=-1) / tracks["diff_time"]
    )
    tracks["speed"] = (tracks["diff_x"] ** 2 + tracks["diff_y"] ** 2) ** 0.5
    tracks["diff_orientation"] = (
        -tracks.groupby("track_id")["orientation"].diff(periods=-1)
        / tracks["diff_time"]
    )
    tracks["diff_axis_1"] = (
        -tracks.groupby("track_id")["axis_1"].diff(periods=-1) / tracks["diff_time"]
    )
    tracks["diff_axis_2"] = (
        -tracks.groupby("track_id")["axis_2"].diff(periods=-1) / tracks["diff_time"]
    )
    tracks = tracks.drop("diff_time", axis=1)

    tracks = tracks.loc[
        np.sqrt(np.square(tracks["x"]) + np.square(tracks["y"])) < MAX_DIST_CENTER
    ]

    all_tracks = []
    track_ids = tracks["track_id"].unique()

    for id in track_ids:
        track = tracks.loc[tracks["track_id"] == id]
        track.index = track["frame"]
        if len(track) > 1:
            track = track.reindex(
                np.arange(track["frame"].min(), track["frame"].max() + 1),
                fill_value=np.nan,
            )
            track["missing"] = track["x"].isna()
            track = track.interpolate()
            track.loc[track["missing"] == True, "confidence"] = np.nan
        all_tracks.append(track)

    tracks = pd.concat(all_tracks, ignore_index=True)
    tracks["track_id"] = tracks["track_id"].astype(np.int64)
    tracks["frame"] = tracks["frame"].astype(np.int64)

    tracks["exp_name"] = exp_name
    tracks["group_name"] = group_name
    tracks["genotype"] = rep_name[0]
    tracks["odor"] = rep_name[1]
    tracks["date"] = pd.to_datetime(rep_name[2], format='mixed', dayfirst=True, errors='coerce')
    tracks["odor_position"] = int(rep_name[3])
    tracks["operator"] = "".join(
        [c for c in rep_name[4] if not c.isdigit()]
    )  # Getting letters
    tracks["operator_nb_video"] = int(
        "".join([c for c in rep_name[4] if c.isdigit()])
    )  # Getting numbers
    tracks["nb_flies"] = int(rep_name[5])

    tracks["dist_center"] = np.sqrt(np.square(tracks["x"]) + np.square(tracks["y"]))

    tracks["scalar_vec_center"] = [
        np.dot(x, y) / np.linalg.norm(y)
        for index, (x, y) in enumerate(
            zip(
                np.column_stack(
                    (
                        tracks["diff_x"],
                        tracks["diff_y"],
                    )
                ),
                np.column_stack((-tracks["x"], -tracks["y"])),
            )
        )
    ]
    tracks["scalar_vec_center_tangent"] = np.sqrt(
        np.abs(np.square(tracks["speed"]) - np.square(tracks["scalar_vec_center"]))
    )

    tracks["vec_odor_x"] = coord_odor[0] - tracks["x"]
    tracks["vec_odor_y"] = coord_odor[1] - tracks["y"]

    tracks["dist_odor"] = np.sqrt(
        np.square(tracks["vec_odor_x"]) + np.square(tracks["vec_odor_y"])
    )

    tracks["scalar_vec_odor"] = [
        np.dot(x, y) / np.linalg.norm(y)
        for index, (x, y) in enumerate(
            zip(
                np.column_stack(
                    (
                        tracks["diff_x"],
                        tracks["diff_y"],
                    )
                ),
                np.column_stack((tracks["vec_odor_x"], tracks["vec_odor_y"])),
            )
        )
    ]

    tracks["scalar_vec_odor_tangent"] = np.sqrt(
        np.abs(np.square(tracks["speed"]) - np.square(tracks["scalar_vec_odor"]))
    )

    with gzip.open(DATAFRAME_FILE, "wt") as f:
        tracks.to_json(f, orient="table")
