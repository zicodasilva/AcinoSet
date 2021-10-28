import os
from typing import Tuple, Dict, Union
import numpy as np

from lib import misc, app
from lib.models import PoseModel, MotionModel
from py_utils import log, data_ops

logger = log.logger(__name__)


def run(init_fname: str,
        n_comps: int = 9,
        window_size: int = 5,
        window_time: int = 2,
        inc_obj_vel: bool = True,
        generative: bool = True,
        plot: bool = False):
    root_dim = 6
    idx = misc.get_pose_params()
    P = len(list(idx.keys()))
    if inc_obj_vel:
        for state in list(idx.keys())[:root_dim]:
            idx["d" + state] = P + idx[state]
    test = data_ops.load_pickle(init_fname)
    pose_model = PoseModel("/Users/zico/msc/data/CheetahRuns/v4/model/dataset_pose.h5",
                           pose_params=idx,
                           ext_dim=root_dim,
                           n_comps=n_comps,
                           standardise=True)
    motion_model = MotionModel("/Users/zico/msc/data/CheetahRuns/v4/model/dataset_runs.h5",
                               len(idx.keys()),
                               window_size=window_size,
                               window_time=window_time,
                               pose_model=pose_model)

    traj_gen = []
    init_traj = test["x"]
    if inc_obj_vel:
        init_traj = np.concatenate((init_traj, test["dx"][:, 0:6]), axis=1)
    gt_pose = pose_model.project(init_traj)
    # Initialise the state by taken the window size number of states to predict the next.
    offset = 10
    init_state = gt_pose[offset:offset + (window_size * window_time)]
    traj_gen = [state for state in init_state]
    # Predict the next state from the initial state and append to the trajectory.
    x_next = _next_state(motion_model, init_state[::window_time])
    traj_gen.append(x_next)
    for i in range(1, len(gt_pose) - window_size * window_time):
        if generative:
            x_prev = np.asarray(traj_gen[-window_size * window_time::window_time])
            x_next = _next_state(motion_model, x_prev)
        else:
            # If we want to use the GT trajectory to estimate the next state at each timestep.
            x_next = _next_state(motion_model, gt_pose[i:i + window_size * window_time:window_time])
        traj_gen.append(x_next)

    # Generate 3D positions from pose parameters and save.
    traj_gen = np.asarray(traj_gen[window_size * window_time:])
    traj = pose_model.project(traj_gen, inverse=True)

    positions = np.asarray([misc.get_3d_marker_coords(pose) for pose in traj])
    saved_traj = {"x": traj, "positions": positions}
    out_fname = os.path.join(os.path.dirname(init_fname), "synthesized_trajectory.pickle")
    data_ops.save_pickle(os.path.join(os.path.dirname(init_fname), "synthesized_trajectory.pickle"), saved_traj)

    # Plot trajectory.
    if plot:
        dummy_scene = "/Users/zico/OneDrive - University of Cape Town/CheetahReconstructionResults/cheetah_videos/2017_08_29/top/extrinsic_calib/6_cam_scene_sba.json"
        app.plot_cheetah_reconstruction(out_fname,
                                        scene_fname=dummy_scene,
                                        reprojections=False,
                                        dark_mode=True,
                                        centered=False)


# Predict the next state from the input vector (window_size x num_vars)
def _next_state(model: MotionModel, x: np.ndarray) -> np.ndarray:
    return model.predict(x) + np.random.default_rng().normal(0, np.sqrt(model.error_variance), x.shape[1])


if __name__ == "__main__":
    test_file = "/Users/zico/msc/dev/AcinoSet/data/2017_08_29/top/jules/run1_1/sd_fte/fte.pickle"
    run(test_file, n_comps=9, window_size=10, window_time=1, plot=True, generative=True, inc_obj_vel=False)
