# Flytracking

**flytracking** is a Python module designed to detect and track flies walking inside an arena. It leverages advanced image processing and machine learning techniques to provide accurate detection and tracking capabilities.

## Features

- **Fly Detection**: Robust detection using image segmentation.
- **Multi-Fly Tracking**: Track multiple flies within an arena using a Kalman filter.
- **Tracklet Connection**: Connect tracklets using global path-finding optimization.
- **Visualization**: Save data, plot summary statistics, and generate detection/tracking videos.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/DatInsights-com/fly_tracking.git
cd flytracking
pip install .
```

## Usage

### 1. Automated Workflow (Recommended)

To run the entire analysis pipeline on a single video file:

```bash
python run_fly_tracker.py VIDEO_PATH OUTPUT_DIR
```

### 2. Step-by-Step Analysis

You can also run individual steps independently using the provided Python functions. These are typically called within a Python script or Jupyter notebook:

```python
from flytracking import (
    create_background_image,
    detect_moving_objects,
    track_moving_objects,
    connect_tracklets,
    plot_longtracks_summary
)

# Example usage
create_background_image(
    video_path="input/video.mp4", 
    out_dir="output", 
    nb_bg_frames=1000, 
    total_frames=18000, 
    overwrite=False
)

detect_moving_objects(
    video_path="input/video.mp4", 
    out_dir="output", 
    total_frames=18000, 
    overwrite=False
)

# Continue with track_moving_objects...
```

### 3. Generating Videos

You can generate visualization movies for your results:

```python
from flytracking import (
    save_detection_movie,
    save_tracking_movie,
    save_longtracks_movie
)

save_detection_movie(
    video_path="input/video.mp4", 
    out_dir="output", 
    total_frames=18000, 
    overwrite=False
)
```

### 4. HPC / Cluster Submission

To process a directory of videos on an HPC cluster, you can generate a batch submission script:

```bash
python generate_slurm_jobs.py VIDEO_DIR OUTPUT_DIR
sh run_all_jobs_VIDEO_DIR.sh
```

## Outputs

The pipeline generates the following outputs in `OUTPUT_DIR`:

- **Coordinates & IDs**: For each fly and frame, saved in JSON format (as a list of lists per frame).
- **Traces & Summaries**: Plots saved as PNG files.
- **Movies**: Visualization of detection and tracking overlaid on the original video.
