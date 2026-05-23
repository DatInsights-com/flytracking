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
MAX_REL_PERIOD_MSD = 0.5

TIME_RESOLUTION_VAC = int(2)
MAX_REL_PERIOD_VAC = 0.1

MIN_RELATIVE_LENGTH = 0.25
MOVING_AVG_WINDOW = int(6000)


matplotlib.use("qtagg")


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

    results = results.sort_values(["temp_id", "frame"])
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

    results = results.drop("temp_id", axis=1)
    return results


def calculate_msd_lontracks(video_path: str, out_dir: str):

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
                0, int(MAX_REL_PERIOD_MSD * max(results["frame"])), TIME_RESOLUTION_MSD
            ),
        ),
        columns=["track_id", "period"],
    )
    calculations["msd"] = np.nan

    for id in results["track_id"].unique():
        data = results.loc[results["track_id"] == id]
        data = data.sort_values(["frame"])
        calculations.loc[
            (calculations["track_id"] == id) & (calculations["period"] == 0), "msd"
        ] = 0

        for p in range(
            TIME_RESOLUTION_MSD,
            min(
                calculations["period"].max(), data["frame"].max() - data["frame"].min()
            ),
            TIME_RESOLUTION_MSD,
        ):
            msd = data["x"].diff(periods=p).pow(2) + data["y"].diff(periods=p).pow(2)
            calculations.loc[
                (calculations["track_id"] == id) & (calculations["period"] == p), "msd"
            ] = msd.mean()

    return calculations


def calculate_vac_lontracks(video_path: str, out_dir: str):

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
                        MAX_REL_PERIOD_VAC * results["frame"].max(),
                        1 / TIME_RESOLUTION_VAC,
                    )
                ),
            ),
        ),
        columns=["track_id", "period"],
    )
    calculations["period"] = calculations["period"].pow(TIME_RESOLUTION_VAC)
    calculations["vac"] = np.nan

    for id in results["track_id"].unique():
        data = results.loc[results["track_id"] == id]
        data = data.sort_values(["frame"])
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
            method_args={"window": min(MOVING_AVG_WINDOW, total_frames)},
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

    p1.save(TRACKING_PLOTS, width=8.268, height=11.693, dpi=600, verbose=False)
    plot.save(SUMMARY_PLOTS, dpi=600, verbose=False)
