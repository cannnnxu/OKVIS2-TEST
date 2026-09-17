import os
from pathlib import Path

import numpy as np
from evo.core import metrics, trajectory, sync
from evo.tools import file_interface

# Paths are configurable so this runs on any machine:
#   OKVIS_OUTPUT    where okvis writes its trajectory csv files (default: <repo>/output)
#   OKVIS_DATASETS  where the datasets live                     (default: ~/Datasets)
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("OKVIS_OUTPUT", REPO_ROOT / "output"))
DATASET_ROOT = Path(os.environ.get("OKVIS_DATASETS", Path.home() / "Datasets"))

def calculate_ate(gt_data, est_data, ground_timestamp, est_timestamp, gps=False):
    traj_ref = trajectory.PoseTrajectory3D(timestamps=ground_timestamp, positions_xyz=gt_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    
    if gps:
        traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    else:
        traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=est_data[:, [6, 3,4, 5]])

    max_diff = 0.001 # Max allowed timestamp difference
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
    traj_est.align(traj_ref, correct_scale=False, correct_only_scale=False)

    # plot the trajectory
    import matplotlib.pyplot as plt
   
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))


    stats = ape_metric.get_all_statistics()
    print(f"ATE RMSE: {stats['rmse']:.4f} m")
    return stats['rmse']

def calculate_rpe(gt_data, est_data, ground_timestamp, est_timestamp):
    traj_ref = trajectory.PoseTrajectory3D(timestamps=ground_timestamp, positions_xyz=gt_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=est_data[:, [6, 3,4, 5]])
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, 0.01)

    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, 
                             delta=1.0, 
                             delta_unit=metrics.Unit.frames,
                             all_pairs=False) 
    
    rpe_metric.process_data((traj_ref, traj_est))
    
    stats = rpe_metric.get_all_statistics()
    print(f"RPE RMSE: {stats['rmse']:.4f}")
    return stats['rmse']

def main(folder, data_mode, data_depth, data_name, datasets):
    est_path = OUTPUT_ROOT / 'cloud_euroc' / data_name / 'super' / '1' / 'okvis2-vio-final-ba_trajectory.csv'
    seq = datasets[data_name]
    gt_path = (DATASET_ROOT / 'EuRoC-Dataset' / 'machine_hall' / seq / seq /
               'mav0' / 'state_groundtruth_estimate0' / 'data.csv')

    est_traj1 = np.loadtxt(est_path, delimiter=',', skiprows=1, usecols=range(10))
    groundtruth = np.loadtxt(gt_path, delimiter=',', skiprows=1, usecols=range(8))
    print(gt_path)
    start_time = est_traj1[0,0] 
    end_time = est_traj1[-1,0] 
    groundtruth = groundtruth[(groundtruth[:,0] >= start_time) & (groundtruth[:,0] <= end_time)]
    est_se31 = est_traj1[:,1:8]
    ground_se3 = groundtruth[:,1:8][:, [0, 1, 2, 4, 5, 6, 3]]
    est_timestamp1 = est_traj1[:,0]

    ground_timestamp = groundtruth[:,0]

    means = 0.0
    print("=================== ATE FINAL ===================")
    # calculate_ate(ground_se3, gps_traj[:,1:8], ground_timestamp, gps_traj[:,0], gps= True)
    calculate_ate(ground_se3, est_se31, ground_timestamp, est_timestamp1)



if __name__ == "__main__":
    # Example usage

    # raw_euroc/MH_03/super/1
    folder = 'cloud'
    data_mode = 'super'
    data_depth = "super"
    data_name = 'MH_01'
    datasets = {
        "MH_01": "MH_01_easy",
        "MH_02": "MH_02_easy",
        "MH_03": "MH_03_medium",
        "MH_04": "MH_04_difficult",
        "MH_05": "MH_05_difficult",
    }
    # main(folder, data_mode, data_depth, data_name, datasets)
    for data_name in datasets.keys():
        # if data_name != "MH_05":
        #     continue
        print(f"=================== {data_name} ===================")
        main(folder, data_mode, data_depth, data_name, datasets)
