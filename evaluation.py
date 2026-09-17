import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pypose as pp

# Paths are configurable so this runs on any machine:
#   OKVIS_OUTPUT    where okvis writes its trajectory csv files (default: <repo>/output)
#   OKVIS_DATASETS  where the datasets live                     (default: ~/Datasets)
#   SEQUENCE        GrAco sequence to evaluate                  (default: aerial02)
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("OKVIS_OUTPUT", REPO_ROOT / "output"))
DATASET_ROOT = Path(os.environ.get("OKVIS_DATASETS", Path.home() / "Datasets"))
SEQUENCE = os.environ.get("SEQUENCE", "aerial02")
GT_SEQUENCE = SEQUENCE.replace("aerial", "aerial-")

est_path = OUTPUT_ROOT / SEQUENCE / 'okvis2-vio-final-ba_trajectory.csv'
gt_path = DATASET_ROOT / 'GrAco' / GT_SEQUENCE / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'

est_traj = torch.from_numpy(np.loadtxt(est_path, delimiter=',', skiprows=1))
groundtruth = torch.from_numpy(np.loadtxt(gt_path, delimiter=' ', skiprows=1))

start_time = est_traj[0,0]
end_time = est_traj[-1,0]
groundtruth = groundtruth[(groundtruth[:,0] >= start_time) & (groundtruth[:,0] <= end_time)]
est_se3 = pp.SE3(est_traj[:,1:8])
ground_se3 = pp.SE3(groundtruth[:,1:8][:, [0, 1, 2, 4, 5, 6, 3]])


# est_pos = ground_se3[0].rotation() * est_se3[0].rotation().Inv() * est_se3.translation()
# est_pos = (ground_se3[0] * est_se3[0].Inv() * est_se3).translation()
est_pos = est_traj[:, 1:]

# baseline_se3 = pp.SE3(baseline_traj[:,1:8])
# baseline_pos = (ground_se3[0] * baseline_se3[0].Inv() * baseline_se3).translation()


# ground_se3[:, :3] = ground_se3[:, :3] - ground_se3[0,:3]
# plot 3d trajectory
ax = plt.figure().add_subplot(projection='3d')
ax.plot(est_pos.numpy()[:, 0], est_pos.numpy()[:, 1], est_pos.numpy()[:, 2], label='Estimated')
# ax.plot(baseline_pos.numpy()[:, 0], baseline_pos.numpy()[:, 1], baseline_pos.numpy()[:, 2], label='Baseline')
# ax.plot(global_pos[:, 1], global_pos[:, 2], global_pos[:, 3], label='Global Position')  # pyright: ignore[reportUndefinedVariable]
ax.plot(ground_se3.translation().numpy()[:, 0], ground_se3.translation().numpy()[:, 1], ground_se3.translation().numpy()[:, 2], label='Ground Truth')
ax.legend()
plt.show()

# # plot 2d trajectory
plt.figure()
plt.plot(est_pos.numpy()[:, 0], est_pos.numpy()[:, 1], label='Estimated')
# plt.plot(baseline_pos.numpy()[:, 0], baseline_pos.numpy()[:, 1], label='Baseline')
# plt.plot(global_pos[:, 1], global_pos[:, 2], label='Global Position')
plt.plot(ground_se3.translation().numpy()[:, 0], ground_se3.translation().numpy()[:, 1], label='Ground Truth')
plt.legend()
plt.show()