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
    # 1. Load trajectories (TUM format example)
    # timestamp x y z q_x q_y q_z q_w
    # traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    # traj_est = file_interface.read_tum_trajectory_file(est_file)
    # positions_xyz: np.ndarray | None = None,
    #     orientations_quat_wxyz
    traj_ref = trajectory.PoseTrajectory3D(timestamps=ground_timestamp, positions_xyz=gt_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    
    if gps:
        traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    else:
        traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=est_data[:, [6, 3,4, 5]])

    # 2. Sync timestamps (match timestamps between Ground Truth and Estimate)
    # Uses nearest neighbor matching by default
    max_diff = 0.001 # Max allowed timestamp difference
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
#     import torch
#     import pypose as pp
#     sync_est = torch.from_numpy(np.concatenate([traj_est.positions_xyz, traj_est.orientations_quat_wxyz[:, [1, 2, 3, 0]]], axis=1))
#     sync_ref = torch.from_numpy(np.concatenate([traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz[:, [1, 2, 3, 0]]], axis=1))

#     sync_est = pp.SE3(sync_est)
#     sync_ref = pp.SE3(sync_ref)
#     sync_est = sync_ref[0] * sync_est[0].Inv() * sync_est
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(sync_est.translation().numpy()[:, 0], sync_est.translation().numpy()[:, 1], label='Estimated')
#     plt.plot(sync_ref.translation().numpy()[:, 0], sync_ref.translation().numpy()[:, 1], label='Ground Truth')
#     plt.legend()
#     plt.show()
#     # ATE
#    #  ROOT MEAN SQUARE OF ABSOLUTE TRAJECTORY ERROR 
#     ate_rmse = np.sqrt(np.mean(np.sum((sync_est.translation().numpy() - sync_ref.translation().numpy())**2, axis=1)))
#     print(f"ATE RMSE: {ate_rmse:.4f} m")
#     return ate_rmse


    # 3. Align trajectories (SE3 Umeyama alignment)
    # This aligns the estimate to the reference
    traj_est.align(traj_ref, correct_scale=False, correct_only_scale=False)

    # plot the trajectory
    import matplotlib.pyplot as plt
    
    # if gps:
    #     plt.figure()
    #     plt.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], label='Ground Truth')
    #     plt.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], label='GPS')
    #     # plot start points
    #     plt.scatter(traj_ref.positions_xyz[0,0], traj_ref.positions_xyz[0,1])
    # else:
    #     # plt.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, 1], label='Ground Truth')
    #     plt.plot(traj_est.positions_xyz[:, 0], traj_est.positions_xyz[:, 1], label='VIO')
    #     plt.legend()
    #     plt.show()
    #     breakpoint()


    # 4. Calculate APE (Absolute Pose Error) - Translation part
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))

    # 5. Get Statistics
    stats = ape_metric.get_all_statistics()
    print(f"ATE RMSE: {stats['rmse']:.4f} m")
    # print(f"ATE Mean: {stats['mean']:.4f} m")
    # print(f"ATE Min:  {stats['min']:.4f} m")
    # print(f"ATE Max:  {stats['max']:.4f} m")
    
    return stats['rmse']

def calculate_rpe(gt_data, est_data, ground_timestamp, est_timestamp, gps=False):
    # 1. Load & Sync (Same as above)
    if gps:
        traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    else:
        traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=est_data[:, [6, 3,4, 5]])

    traj_ref = trajectory.PoseTrajectory3D(timestamps=ground_timestamp, positions_xyz=gt_data[:, :3], orientations_quat_wxyz=gt_data[:, [6, 3,4, 5]])
    # traj_est = trajectory.PoseTrajectory3D(timestamps=est_timestamp, positions_xyz=est_data[:, :3], orientations_quat_wxyz=est_data[:, [6, 3,4, 5]])
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, 0.01)

    # 2. Calculate RPE (Relative Pose Error)
    # delta=1, delta_unit=metrics.Unit.frames (drift per frame)
    # For drift per meter, you need more complex setup usually handled by CLI
    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, 
                             delta=1.0, 
                             delta_unit=metrics.Unit.frames,
                             all_pairs=True) 
    
    rpe_metric.process_data((traj_ref, traj_est))
    
    stats = rpe_metric.get_all_statistics()
    print(f"RPE RMSE: {stats['rmse']:.4f}")
    return stats['rmse']

def main(folder, data_mode, data_depth, data_name, datasets):
    est_path = OUTPUT_ROOT / data_name / 'superpoints' / 'okvis2-vio-final-ba_trajectory.csv'
    gt_path = DATASET_ROOT / 'GrAco' / datasets[data_name] / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
    gps_path = OUTPUT_ROOT / data_name / 'gps' / 'okvis2-vio-global-final-ba_trajectory.csv'

    est_traj1 = np.loadtxt(est_path, delimiter=',', skiprows=1, usecols=range(10))
    groundtruth = np.loadtxt(gt_path, delimiter=' ', skiprows=1, usecols=range(8))
    gps_traj = np.loadtxt(gps_path, delimiter=',', skiprows=1, usecols=range(4))
    print(gt_path)
    start_time = est_traj1[0,0] 
    end_time = est_traj1[-1,0] 
    groundtruth = groundtruth[(groundtruth[:,0] >= start_time) & (groundtruth[:,0] <= end_time)]
    est_se31 = est_traj1[:,1:8]

    ground_se3 = groundtruth[:,1:8][:, [0, 1, 2, 4, 5, 6, 3]]
    est_timestamp1 = est_traj1[:,0]

    ground_timestamp = groundtruth[:,0]

    # est_se31_depth = est_traj1_depth[:,1:8]

    means = 0.0
    print("=================== ATE SUPER ===================")
    calculate_ate(ground_se3, gps_traj[:,1:8], ground_timestamp, gps_traj[:,0], gps= True)

if __name__ == "__main__":
    # Example usage

    # raw_euroc/MH_03/super/1
    folder = 'cloud'
    data_mode = 'super'
    data_depth = "super"
    # data_name = 'aerial01'
    datasets = {
        "aerial01": "aerial-01",
        "aerial02": "aerial-02",
        "aerial03": "aerial-03",
        "aerial04": "aerial-04",
        "aerial05": "aerial-05",
    }
    # main(folder, data_mode, data_depth, data_name, datasets)
    for data_name in datasets.keys():
        if data_name == "aerial03":
            continue
        print(f"=================== {data_name} ===================")
        main(folder, data_mode, data_depth, data_name, datasets)
