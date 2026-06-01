import os
import glob
import sys
from pathlib import Path


def generate_slurm_job(
    video_path: str,
    out_dir: str,
    cores: int,
    memory: int,
    time: int,
    partition: str,
    project: str,
    user: str,
    process_command: str,
):
    job_name = Path(video_path).stem
    out_dir = Path(out_dir).joinpath(Path(video_path).parent)

    script_content = []

    script_content.append("#!/bin/bash")
    script_content.append(f"#SBATCH --job-name={job_name}")
    script_content.append("#SBATCH --ntasks=1")
    script_content.append(f"#SBATCH --cpus-per-task={cores}")
    script_content.append(f"#SBATCH --mem={memory}GB")
    script_content.append(f"#SBATCH --time={time}:00:00")
    script_content.append(f"#SBATCH --partition={partition}")
    script_content.append("#SBATCH --output=logs/%x_%j.out")
    script_content.append("#SBATCH --error=logs/%x_%j.err")
    script_content.append("")

    script_content.append("module purge")
    script_content.append(f"source /fshpc/{user}/.bashrc")
    script_content.append("conda_initialize")
    script_content.append(
        "micromamba activate /lustre/project/ki-flytrack/env_fly_tracking"
    )
    script_content.append("")

    script_content.append(f"python {process_command} {video_path} {out_dir}")

    slurm_job = f"jobs/job_{job_name}.sh"

    with open(slurm_job, "w") as f:
        f.write("\n".join(script_content))

    return slurm_job


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python generate_slurm_jobs.py video_dir output_dir ")
        sys.exit(1)

    # Configure your paths and commands here
    VIDEO_DIR = sys.argv[1]
    OUT_DIR = sys.argv[2]

    dir_name = Path(VIDEO_DIR).name
    SCRIPT_NAME = f"run_all_jobs_{dir_name}.sh"

    PROCESS_COMMAND = "/lustre/project/ki-flytrack/run_fly_tracker.py"

    N_CORES = 4
    MEMORY_GB = 8
    TIME_H = 1
    PARTITION = "ki-smallcpu"
    PROJECT = "ki-flytrack"
    USER = os.getlogin()

    video_files = list(Path(VIDEO_DIR).rglob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"Found {len(video_files)} .mp4 files.")

    all_jobs = []
    all_jobs.append("#!/bin/bash")

    for video in video_files:
        job = generate_slurm_job(
            video_path=video,
            out_dir=OUT_DIR,
            cores=N_CORES,
            memory=MEMORY_GB,
            time=TIME_H,
            partition=PARTITION,
            project=PROJECT,
            user=USER,
            process_command=PROCESS_COMMAND,
        )
        all_jobs.append(f"sbatch {job}")

    all_jobs.append("")

    with open(SCRIPT_NAME, "w") as f:
        f.write("\n".join(all_jobs))
