import os
import platform
import json
from typing import Union, Tuple, Dict
import numpy as np
from pyomo.core.expr.current import log as pyomo_log, exp
import sympy as sp
import pandas as pd
from glob import glob
from time import time
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from lib import misc, utils, app
from lib.calib import project_points_fisheye, triangulate_points_single_img
from lib.models import PoseModel, MotionModel, PoseModelGMM
from scipy.stats import multivariate_normal

from py_utils import data_ops, log

# Create a module logger with the name of this file.
logger = log.logger(__name__)


def plot_init_traj(out_dir: str, x_points: np.ndarray, y_points: np.ndarray):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 12), dpi=120)
    plt.plot(x_points, y_points)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim((0, 12))
    fig.savefig(os.path.join(out_dir, "init_traj.png"))
    plt.close()


def traj_error(X: np.ndarray, Y: np.ndarray, centered: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    X = np.asarray(X)
    Y = np.asarray(Y)
    markers = misc.get_markers()
    if centered:
        X -= np.expand_dims(np.mean(X, axis=1), axis=1)
        Y -= np.expand_dims(np.mean(Y, axis=1), axis=1)
    distances = np.sqrt(np.sum((X - Y)**2, axis=2))
    trajectory_error_mm = np.mean(distances, axis=1) * 1000.0
    mpjpe_mm = np.mean(distances, axis=0) * 1000.0
    result = pd.DataFrame(mpjpe_mm.reshape(1, len(markers)), columns=markers)
    logger.info(f"Single view error [mm] across the entire trajectory: {float(result.mean(axis=1))}")
    result = result.transpose()
    result.columns = ["mpjpe (mm)"]

    return result.astype(float), trajectory_error_mm


def compare_traj_error(fte_orig: str,
                       fte: str,
                       root_dir: str,
                       data_dir: str,
                       fte_type: str = "sd_fte",
                       out_dir_prefix: str = None) -> None:
    import matplotlib.pyplot as plt

    # Gather data for the mult-view, single-view and single view with contraints.
    if out_dir_prefix:
        fte_multi_view = os.path.join(out_dir_prefix, data_dir, fte_type, "fte.pickle")
    else:
        fte_multi_view = os.path.join(root_dir, data_dir, fte_type, "fte.pickle")
    multi_view_data = data_ops.load_pickle(fte_multi_view)
    single_view_data = data_ops.load_pickle(fte_orig)
    pose_model_data = data_ops.load_pickle(fte)
    single_view_mpjpe_mm, single_view_error = traj_error(multi_view_data["positions"], single_view_data["positions"])
    _, pose_model_error = traj_error(multi_view_data["positions"], pose_model_data["positions"])

    # Plot the error between the single view and snigle view with constraints.
    fig = plt.figure(figsize=(16, 12), dpi=120)
    plt.plot(single_view_error, label="single-view")
    plt.plot([np.mean(single_view_error)] * len(single_view_error), label="mean single-view", linestyle="--")
    plt.plot(pose_model_error, label="single-view with pose contraints")
    plt.plot([np.mean(pose_model_error)] * len(pose_model_error),
             label="mean single-view with pose contraints",
             linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Error (mm)")
    ax = plt.gca()
    ax.legend()
    fig.savefig(os.path.join(os.path.dirname(fte), "traj_error.png"))
    plt.cla()

    # Plot the X and Y positions of the single view with constraints for the trajectory.
    pts_3d_spine_multi = np.asarray(multi_view_data["positions"])[:, 4, :]
    pts_3d_spine_orig = np.asarray(single_view_data["positions"])[:, 4, :]
    pts_3d_spine = np.asarray(pose_model_data["positions"])[:, 4, :]
    # fig = plt.figure()
    plt.plot(pts_3d_spine_multi[:, 0], pts_3d_spine_multi[:, 1], label="multi-view")
    plt.plot(pts_3d_spine_orig[:, 0], pts_3d_spine_orig[:, 1], label="single-view")
    plt.plot(pts_3d_spine[:, 0], pts_3d_spine[:, 1], label="single-view with pose constraints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim((0, 12))
    ax = plt.gca()
    ax.legend()
    fig.savefig(os.path.join(os.path.dirname(fte), "x_y_positions.png"))
    plt.cla()

    ax = single_view_mpjpe_mm["mpjpe (mm)"].plot(kind="barh")
    fig = ax.get_figure()
    plt.xlabel("Error (mm)")
    plt.ylabel("Bodypart")
    fig.savefig(os.path.join(os.path.dirname(fte), "mpjpe_dist.png"))
    plt.cla()

    # Plot the hip and shoulder states to get an idea of how well the gait is tracked.
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
    axs = axs.flatten()
    p_idx = misc.get_pose_params()
    x_gt = multi_view_data["x"]
    x = pose_model_data["x"]
    states = ("theta_6", "theta_8", "theta_10", "theta_12")
    titles = ("Left Shoulder Angle", "Right Shoulder Angle", "Left Hip Angle", "Right Hip Angle")
    for i, state in enumerate(states):
        axs[i].plot(x_gt[:, p_idx[state]], label="GT multi-view")
        axs[i].plot(x[:, p_idx[state]], label="single-view")
        axs[i].set_title(titles[i])
        axs[i].legend()
    fig.savefig(os.path.join(os.path.dirname(fte), "states_comparison.png"))


def get_vals_v(var: Union[pyo.Var, pyo.Param], idxs: list) -> np.ndarray:
    """
    Verbose version that doesn't try to guess stuff for ya. Usage:
    >>> get_vals(m.q, (m.N, m.DOF))
    """
    arr = np.array([pyo.value(var[idx]) for idx in var]).astype(float)
    return arr.reshape(*(len(i) for i in idxs))


def pyo_i(i: int) -> int:
    return i + 1


def plot_trajectory(fte_file: str, scene_file: str, centered=False):
    app.plot_cheetah_reconstruction(fte_file,
                                    scene_fname=scene_file,
                                    reprojections=False,
                                    dark_mode=True,
                                    centered=centered)


def compare_trajectories(fte_file1: str, fte_file2: str, scene_file: str, centered=False):
    app.plot_multiple_cheetah_reconstructions([fte_file1, fte_file2],
                                              scene_fname=scene_file,
                                              reprojections=False,
                                              dark_mode=True,
                                              centered=centered)


def plot_cheetah(root_dir: str,
                 data_dir: str,
                 fte_type: str = "pw_sd_fte",
                 out_dir_prefix: str = None,
                 plot_reprojections=False,
                 centered=False):
    fte_file = os.path.join(root_dir, data_dir, fte_type, "fte.pickle")
    *_, scene_fpath = utils.find_scene_file(os.path.join(root_dir, data_dir))
    if out_dir_prefix is not None:
        fte_file = os.path.join(out_dir_prefix, data_dir, fte_type, "fte.pickle")
    app.plot_cheetah_reconstruction(fte_file,
                                    scene_fname=scene_fpath,
                                    reprojections=plot_reprojections,
                                    dark_mode=True,
                                    centered=centered)


def compare_cheetahs(test_fte_file: str,
                     root_dir: str,
                     data_dir: str,
                     fte_type: str = "pw_sd_fte",
                     out_dir_prefix: str = None,
                     plot_reprojections=False,
                     centered=False):
    fte_file = os.path.join(root_dir, data_dir, fte_type, "fte.pickle")
    *_, scene_fpath = utils.find_scene_file(os.path.join(root_dir, data_dir))
    if out_dir_prefix is not None:
        fte_file = os.path.join(out_dir_prefix, data_dir, fte_type, "fte.pickle")
    app.plot_multiple_cheetah_reconstructions([fte_file, test_fte_file],
                                              scene_fname=scene_fpath,
                                              reprojections=plot_reprojections,
                                              dark_mode=True,
                                              centered=centered)


def plot_cost_functions():
    import matplotlib.pyplot as plt
    # cost function
    redesc_a = 5
    redesc_b = 15
    redesc_c = 30
    r_x = np.arange(-100, 100, 1e-1)
    r_y1 = [misc.redescending_loss(i, redesc_a, redesc_b, redesc_c) for i in r_x]
    r_y2 = [misc.cauchy_loss(i, 7, np.log) for i in r_x]
    r_y3 = r_x**2
    r_y4 = [misc.fair_loss(i, redesc_b, np.log) for i in r_x]
    plt.figure()
    plt.plot(r_x, r_y1, label="Redescending")
    plt.plot(r_x, r_y2, label="Cauchy")
    plt.plot(r_x, r_y4, label="Fair")
    plt.plot(r_x, r_y3, label="LSQ")
    ax = plt.gca()
    ax.set_ylim((-5, 500))
    ax.legend()
    plt.show(block=True)


def loss_function(residual: float, loss="redescending") -> float:
    if loss == "redescending":
        return misc.redescending_loss(residual, 3, 10, 20)
    elif loss == "cauchy":
        return misc.cauchy_loss(residual, 7, pyomo_log)
    elif loss == "fair":
        return misc.fair_loss(residual, 10, pyomo_log)
    elif loss == "lsq":
        return residual**2

    return 0.0


def create_pose_functions(data_dir: str):
    # symbolic vars
    idx = misc.get_pose_params()
    sym_list = sp.symbols(list(idx.keys()))
    positions = misc.get_3d_marker_coords(sym_list)

    func_map = {"sin": pyo.sin, "cos": pyo.cos, "ImmutableDenseMatrix": np.array}
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i, :], modules=[func_map])
        pos_funcs.append(lamb)

    # Save the functions to file.
    data_ops.save_dill(os.path.join(data_dir, "pose_3d_functions_with_paws.pickle"), (pose_to_3d, pos_funcs))


def run(root_dir: str,
        data_path: str,
        cam_idx: int,
        start_frame: int,
        end_frame: int,
        dlc_thresh: float,
        fte_type: str = "sd_fte",
        extra_constraints: bool = True,
        loss="redescending",
        n_comps: int = 5,
        pairwise_included: int = 0,
        reduced_space: bool = False,
        inc_obj_orien: bool = False,
        inc_obj_vel: bool = True,
        use_gmm: bool = True,
        init_fte=False,
        opt=None,
        out_dir_prefix: str = None,
        generate_reprojection_videos: bool = False):
    logger.info("Prepare data - Start")

    t0 = time()

    if out_dir_prefix:
        out_dir = os.path.join(out_dir_prefix, data_path,
                               f"fte_{cam_idx}" if extra_constraints else f"fte_orig_{cam_idx}")
        fte_dir = os.path.join(out_dir_prefix, data_path, f"fte_orig_{cam_idx}")
    else:
        out_dir = os.path.join(root_dir, "cheetah_videos", data_path, f"fte_{cam_idx}" if extra_constraints else f"fte_orig_{cam_idx}")
        fte_dir = os.path.join(root_dir, "cheetah_videos", data_path, f"fte_orig_{cam_idx}")

    if extra_constraints:
        assert os.path.exists(fte_dir)

    data_dir = os.path.join(root_dir, "cheetah_videos", data_path)
    assert os.path.exists(data_dir)
    dlc_dir = os.path.join(data_dir, "dlc")
    assert os.path.exists(dlc_dir)
    os.makedirs(out_dir, exist_ok=True)

    app.start_logging(os.path.join(out_dir, "fte.log"))

    logger.info("Load data")
    # ========= IMPORT CAMERA & SCENE PARAMS ========
    try:
        k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(data_dir)
    except Exception:
        logger.error("Early exit because extrinsic calibration files could not be located")
        return
    d_arr = d_arr.reshape((-1, 4))

    # load video info
    res, fps, num_frames = 0, 0, 0
    if platform.python_implementation() == "CPython":
        res, fps, num_frames, _ = app.get_vid_info(data_dir)  # path to the directory having original videos
        assert res == cam_res
    elif platform.python_implementation() == "PyPy":
        fps = 120 if "2019" in data_dir else 90

    # load DLC data
    dlc_points_fpaths = sorted(glob(os.path.join(dlc_dir, "*.h5")))
    assert n_cams == len(dlc_points_fpaths), f"# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json"

    # Load measurement data.
    df_paths = sorted(glob(os.path.join(dlc_dir, "*.h5")))
    h5_filename = os.path.basename(df_paths[cam_idx])
    pw_data = data_ops.load_pickle(
        os.path.join(f"{dlc_dir}_pw", f"{h5_filename[:4]}DLC_resnet152_CheetahOct14shuffle4_650000.pickle"))
    points_2d_df = pd.read_hdf(os.path.join(dlc_dir, h5_filename))
    base_data = list(points_2d_df.to_numpy())
    # Re-organise data to make for easy selection of data in the dataframe.
    points_2d_df = points_2d_df.droplevel([0], axis=1).swaplevel(0, 1, axis=1).T.unstack().T.reset_index().rename(
        {"level_0": "frame"}, axis=1)
    points_2d_df.columns.name = ""
    points_2d_df.rename(columns={"bodyparts": "marker"}, inplace=True)

    if platform.python_implementation() == "PyPy":
        # At the moment video reading does not work with openCV and PyPy - well at least not on the Linux i9.
        # So instead I manually get the number of frames and the frame rate based (determined above).
        num_frames = points_2d_df["frame"].max() + 1

    assert 0 != start_frame < num_frames, f"start_frame must be strictly between 0 and {num_frames}"
    assert 0 != end_frame <= num_frames, f"end_frame must be less than or equal to {num_frames}"
    assert 0 <= dlc_thresh <= 1, "dlc_thresh must be from 0 to 1"

    if start_frame == -1:
        # Use the entire video.
        start_frame = 1
        end_frame = num_frames
    else:
        # User-defined frames
        start_frame = start_frame - 1  # 0 based indexing
        end_frame = end_frame % num_frames + 1 if end_frame == -1 else end_frame

    N = end_frame - start_frame
    Ts = 1.0 / fps  # timestep
    # Check that we have a valid range of frames - if not perform the entire sequence.
    if N == 0:
        end_frame = num_frames
        N = end_frame - start_frame
    # For memory reasons - do not perform optimisation on trajectories larger than 200 points.
    if N > 200:
        end_frame = start_frame + 200
        N = end_frame - start_frame

    ## ========= POSE FUNCTIONS ========
    pose_to_3d, pos_funcs = data_ops.load_dill(os.path.join(root_dir, "cheetah_videos", "pose_3d_functions_with_paws.pickle"))
    idx = misc.get_pose_params()
    sym_list = list(idx.keys())

    # ========= PROJECTION FUNCTIONS ========
    def pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
        y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
        z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
        #project onto camera plane
        a = x_2d / z_2d
        b = y_2d / z_2d
        #fisheye params
        r = (a**2 + b**2)**0.5
        th = pyo.atan(r)
        #distortion
        th_d = th * (1 + D[0] * th**2 + D[1] * th**4 + D[2] * th**6 + D[3] * th**8)
        x_p = a * th_d / (r + 1e-12)
        y_p = b * th_d / (r + 1e-12)
        u = K[0, 0] * x_p + K[0, 2]
        v = K[1, 1] * y_p + K[1, 2]
        return u, v

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        return pt3d_to_2d(x, y, z, K, D, R, t)[0]

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        return pt3d_to_2d(x, y, z, K, D, R, t)[1]

    # ========= IMPORT DATA ========
    markers = misc.get_markers()
    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    # measurement standard deviation
    R = np.array(
        [
            1.2,  # nose
            1.24,  # l_eye
            1.18,  # r_eye
            2.08,  # neck_base
            2.04,  # spine
            2.52,  # tail_base
            2.73,  # tail1
            1.83,  # tail2
            3.47,  # r_shoulder
            2.75,  # r_front_knee
            2.69,  # r_front_ankle
            2.24,  # r_front_paw
            3.4,  # l_shoulder
            2.91,  # l_front_knee
            2.85,  # l_front_ankle
            2.27,  # l_front_paw
            3.26,  # r_hip
            2.76,  # r_back_knee
            2.33,  # r_back_ankle
            2.4,  # r_back_paw
            3.53,  # l_hip
            2.69,  # l_back_knee
            2.49,  # l_back_ankle
            2.34,  # l_back_paw
        ],
        dtype=float)
    R_pw = np.array([
        R,
        [
            2.71, 3.06, 2.99, 4.07, 5.53, 4.67, 6.05, 5.6, 5.01, 5.11, 5.24, 4.85, 5.18, 5.28, 5.5, 4.9, 4.7, 4.7, 5.21,
            5.11, 5.1, 5.27, 5.75, 5.44
        ],
        [
            2.8, 3.24, 3.42, 3.8, 4.4, 5.43, 5.22, 7.29, 8.19, 6.5, 5.9, 6.18, 8.83, 6.52, 6.22, 6.34, 6.8, 6.12, 5.37,
            5.98, 7.83, 6.44, 6.1, 6.38
        ]
    ],
                    dtype=float)
    R_pw *= 1.5

    Q = np.array([  # model parameters variance
        4,
        7,
        5,  # head position in inertial
        13,
        9,
        26,  # head rotation in inertial
        32,
        18,
        12,  # neck
        43,  # front torso
        10,
        53,
        34,  # back torso
        90,
        43,  # tail_base
        118,  # tail_mid
        51,
        247,  # l_shoulder
        186,  # l_front_knee
        194,  # r_shoulder
        164,  # r_front_knee
        295,  # l_hip
        243,  # l_back_knee
        334,  # r_hip
        149,  # r_back_knee
        91,  # l_front_ankle
        91,  # r_front_ankle
        132,  # l_back_ankle
        132  # r_back_ankle
    ])**2

    # Estimate initial points
    logger.info("Estimate the initial trajectory")
    # Use the cheetahs spine to estimate the initial trajectory with a 3rd degree spline.
    frame_range = np.arange(num_frames, dtype=int)

    points_2d_df.loc[points_2d_df["likelihood"] <= dlc_thresh, ["x", "y"]] = np.nan  # ignore points with low likelihood
    nose_pts_2d = points_2d_df.query("marker == 'nose'")[["x", "y"]].to_numpy(dtype=np.float32)
    X_w = triangulate_points_single_img(nose_pts_2d, 3, k_arr[cam_idx], d_arr[cam_idx], r_arr[cam_idx], t_arr[cam_idx])
    w = np.isnan(X_w[0, :])
    X_w[0, w] = 0.0
    x_est = np.array(UnivariateSpline(frame_range, X_w[0, :], w=~w)(frame_range))
    w = np.isnan(X_w[1, :])
    X_w[1, w] = 0.0
    y_est = np.array(UnivariateSpline(frame_range, X_w[1, :], w=~w)(frame_range))
    w = np.isnan(X_w[2, :])
    X_w[2, w] = 0.0
    z_est = np.array(UnivariateSpline(frame_range, X_w[2, :], w=~w)(frame_range))

    # Plot the initial trajectory on the XY plane to validate the single view initial trajectory is valid.
    plot_init_traj(out_dir, x_est, y_est)

    # Calculate the initial yaw.
    dx_est = np.diff(x_est) / Ts
    dy_est = np.diff(y_est) / Ts
    psi_est = np.arctan2(dy_est, dx_est)
    # Duplicate the last heading estimate as the difference calculation returns N-1.
    psi_est = np.append(psi_est, [psi_est[-1]])

    logger.info("Prepare data - End")
    # save parameters
    with open(os.path.join(out_dir, "reconstruction_params.json"), "w") as f:
        json.dump(dict(start_frame=start_frame, end_frame=end_frame, dlc_thresh=dlc_thresh), f)

    logger.info(f"Start frame: {start_frame}, End frame: {end_frame}, Frame rate: {fps}")

    #===================================================
    #                   Optimisation
    #===================================================
    logger.info("Setup optimisation - Start")
    root_dim = 6
    if inc_obj_orien:
        ext_dim = 4
    else:
        ext_dim = 6
    window_size = 10
    window_time = 1
    window_buf = window_size * window_time
    m = pyo.ConcreteModel(name="Cheetah from measurements")
    m.Ts = Ts
    # ===== SETS =====
    P = len(list(sym_list))  # number of pose parameters
    L = len(markers)  # number of dlc labels per frame

    m.N = pyo.RangeSet(N)
    if reduced_space:
        m.P = pyo.RangeSet(ext_dim + n_comps)
    else:
        m.P = pyo.RangeSet(P)
    m.L = pyo.RangeSet(L)
    # Dimensionality of measurements
    m.D2 = pyo.RangeSet(2)
    m.D3 = pyo.RangeSet(3)
    # Number of pairwise terms to include + the base measurement.
    m.W = pyo.RangeSet(pairwise_included + 1 if pairwise_included <= 2 and pairwise_included >= 0 else 1)

    index_dict = misc.get_dlc_marker_indices()
    pair_dict = misc.get_pairwise_graph()

    # Instantiate the reduced pose model.
    if inc_obj_vel and not reduced_space:
        for state in list(idx.keys())[:root_dim]:
            idx["d" + state] = P + idx[state]
    pose_model = PoseModel(os.path.join(root_dir, "cheetah_runs", "v4", "model", "dataset_pose.h5"),
                           pose_params=idx,
                           ext_dim=ext_dim,
                           n_comps=n_comps,
                           standardise=True)
    pose_var = 0.3 * pose_model.error_variance

    pose_gmm = PoseModelGMM("/Users/zico/msc/data/CheetahRuns/v4/model/dataset_runs.h5",
                            pose_params=idx,
                            ext_dim=ext_dim,
                            n_comps=n_comps)

    def norm_pdf_multivariate(x: np.ndarray, mu: np.ndarray, cov: np.ndarray):
        part1 = 1 / (((2 * np.pi)**(len(mu) / 2)) * (np.linalg.det(cov)**(1 / 2)))
        part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
        return part1 * exp(part2)

    # Train motion model and make predictions with a predefined window size.
    motion_model = MotionModel(os.path.join(root_dir, "cheetah_runs", "v4", "model", "dataset_runs.h5"),
                               P,
                               window_size=window_size,
                               window_time=window_time,
                               lasso=True,
                               pose_model=pose_model if reduced_space else None)
    pred_var = 0.5 * motion_model.error_variance

    if reduced_space:
        Q = np.abs(pose_model.project(Q))
    # ======= WEIGHTS =======
    def init_meas_weights(m, n, l, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        marker = markers[l - 1]
        values = pw_data[(n - 1) + start_frame]
        likelihoods = values["pose"][2::3]
        if w < 2:
            base = index_dict[marker]
            likelihoods = base_data[(n - 1) + start_frame][2::3]
        else:
            try:
                base = pair_dict[marker][w - 2]
            except IndexError:
                return 0.0

        # Filter measurements based on DLC threshold.
        # This does ensures that badly predicted points are not considered in the objective function.
        return 1 / R_pw[w - 1][l - 1] if likelihoods[base] > dlc_thresh else 0.0

    m.meas_err_weight = pyo.Param(m.N, m.L, m.W, initialize=init_meas_weights, mutable=True)
    m.model_err_weight = pyo.Param(m.P, initialize=lambda m, p: 1 / Q[p - 1] if Q[p - 1] != 0.0 else 0.0)
    m.pose_err_weight = pyo.Param(m.P, initialize=lambda m, p: 1 / pose_var[p - 1] if pose_var[p - 1] != 0.0 else 0.0)
    m.motion_err_weight = pyo.Param(m.P, initialize=lambda m, p: 1 / pred_var[p - 1] if pred_var[p - 1] != 0 else 0.0)

    # ===== PARAMETERS =====
    def init_measurements(m, n, l, d2, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        marker = markers[l - 1]
        if w < 2:
            base = index_dict[marker]
            val = base_data[(n - 1) + start_frame][d2 - 1::3]

            return val[base]
        else:
            try:
                values = pw_data[(n - 1) + start_frame]
                val = values["pose"][d2 - 1::3]
                base = pair_dict[marker][w - 2]
                val_pw = values["pws"][:, :, :, d2 - 1]
                return val[base] + val_pw[0, base, index_dict[marker]]
            except IndexError:
                return 0.0

    m.meas = pyo.Param(m.N, m.L, m.D2, m.W, initialize=init_measurements)

    logger.info("Measurement initialisation...Done")
    # ===== VARIABLES =====
    m.x = pyo.Var(m.N, m.P)  #position
    m.dx = pyo.Var(m.N, m.P)  #velocity
    m.ddx = pyo.Var(m.N, m.P)  #acceleration
    m.poses = pyo.Var(m.N, m.L, m.D3)
    m.slack_model = pyo.Var(m.N, m.P, initialize=0.0)
    m.slack_meas = pyo.Var(m.N, m.L, m.D2, m.W, initialize=0.0)
    m.slack_pose = pyo.Var(m.N, m.P, initialize=0.0)
    m.slack_motion = pyo.Var(m.N, m.P, initialize=0.0)

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
    if init_fte:
        fte_states = data_ops.load_pickle(os.path.join(os.path.dirname(out_dir), fte_type, "fte.pickle"))
        init_x = fte_states["x"]
        init_dx = fte_states["dx"]
        init_ddx = fte_states["ddx"]
    else:
        init_x[:, idx["x_0"]] = x_est[start_frame:start_frame + N]  #x # change this to [start_frame: end_frame]?
        init_x[:, idx["y_0"]] = y_est[start_frame:start_frame + N]  #y
        init_x[:, idx["z_0"]] = z_est[start_frame:start_frame + N]  #z
        init_x[:, idx["psi_0"]] = psi_est[start_frame:start_frame + N]  # yaw = psi

    if reduced_space:
        init_x = pose_model.project(init_x)
    for n in m.N:
        for p in m.P:
            if n < len(init_x):  #init using known values
                m.x[n, p].value = init_x[n - 1, p - 1]
                m.dx[n, p].value = init_dx[n - 1, p - 1]
                m.ddx[n, p].value = init_ddx[n - 1, p - 1]
            else:  #init using last known value
                m.x[n, p].value = init_x[-1, p - 1]
                m.dx[n, p].value = init_dx[-1, p - 1]
                m.ddx[n, p].value = init_ddx[-1, p - 1]
        #init pose
        var_list = [m.x[n, p].value for p in m.P]
        if reduced_space:
            var_list = pose_model.project(var_list, inverse=True)
        for l in m.L:
            [pos] = pos_funcs[l - 1](*var_list)
            for d3 in m.D3:
                m.poses[n, l, d3].value = pos[d3 - 1]

    logger.info("Variable initialisation...Done")

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m, n, l, d3):
        # Get 3d points
        var_list = [m.x[n, p] for p in m.P]
        if reduced_space:
            var_list = pose_model.project(var_list, inverse=True)
        [pos] = pos_funcs[l - 1](*var_list)
        return pos[d3 - 1] == m.poses[n, l, d3]

    m.pose_constraint = pyo.Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # INTEGRATION
    def backwards_euler_pos(m, n, p):  # position
        return m.x[n, p] == m.x[n - 1, p] + m.Ts * m.dx[n, p] if n > 1 else pyo.Constraint.Skip

    m.integrate_p = pyo.Constraint(m.N, m.P, rule=backwards_euler_pos)

    def backwards_euler_vel(m, n, p):  # velocity
        return m.dx[n, p] == m.dx[n - 1, p] + m.Ts * m.ddx[n, p] if n > 1 else pyo.Constraint.Skip

    m.integrate_v = pyo.Constraint(m.N, m.P, rule=backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p):
        return m.ddx[n, p] == m.ddx[n - 1, p] + m.slack_model[n, p] if n > 1 else pyo.Constraint.Skip

    m.constant_acc = pyo.Constraint(m.N, m.P, rule=constant_acc)

    if extra_constraints:

        def motion_constraint(m, n, p):
            if n > window_buf:
                # Pred using LR model
                X = [[m.x[n - w * window_time, i] for i in m.P] for w in range(window_size, 0, -1)]
                y = motion_model.predict(X).tolist()
                return m.x[n, p] - y[p - 1] - m.slack_motion[n, p] == 0.0
            else:
                return pyo.Constraint.Skip

        m.motion_constraint = pyo.Constraint(m.N, m.P, rule=motion_constraint)

        if not reduced_space and not use_gmm:

            def reduced_pose_constraint(m, n, p):
                # Contrain poses that are close to the reduced pose space.
                x = [m.x[n, i] for i in range(1, P + 1)]
                if inc_obj_vel:
                    x += [m.dx[n, i] for i in range(1, root_dim + 1)]
                x_r = pose_model.project(pose_model.project(x), inverse=True).tolist()
                return m.x[n, p] - x_r[p - 1] - m.slack_pose[n, p] == 0.0

            m.reduced_pose_constraint = pyo.Constraint(m.N, m.P, rule=reduced_pose_constraint)

    # MEASUREMENT
    def measurement_constraints(m, n, l, d2, w):
        #project
        K, D, R, t = k_arr[cam_idx], d_arr[cam_idx], r_arr[cam_idx], t_arr[cam_idx]
        x = m.poses[n, l, 1]
        y = m.poses[n, l, 2]
        z = m.poses[n, l, 3]

        return proj_funcs[d2 - 1](x, y, z, K, D, R, t) - m.meas[n, l, d2, w] - m.slack_meas[n, l, d2, w] == 0.0

    m.measurement = pyo.Constraint(m.N, m.L, m.D2, m.W, rule=measurement_constraints)

    #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
    # Head
    m.head_phi_0 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["phi_0"])], np.pi / 6))
    m.head_theta_0 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["theta_0"])], np.pi / 6))
    if not reduced_space:
        # Neck
        m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 2, m.x[n, pyo_i(idx["phi_1"])], np.pi / 2))
        m.neck_theta_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["theta_1"])], np.pi / 6))
        m.neck_psi_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["psi_1"])], np.pi / 6))
        # Front torso
        m.front_torso_theta_2 = pyo.Constraint(m.N,
                                               rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["theta_2"])], np.pi / 6))
        # Back torso
        m.back_torso_theta_3 = pyo.Constraint(m.N,
                                              rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["theta_3"])], np.pi / 6))
        m.back_torso_phi_3 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["phi_3"])], np.pi / 6))
        m.back_torso_psi_3 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, pyo_i(idx["psi_3"])], np.pi / 6))
        # Tail base
        m.tail_base_theta_4 = pyo.Constraint(m.N,
                                             rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, pyo_i(idx["theta_4"])],
                                                                (2 / 3) * np.pi))
        m.tail_base_psi_4 = pyo.Constraint(m.N,
                                           rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, pyo_i(idx["psi_4"])],
                                                              (2 / 3) * np.pi))
        # Tail mid
        m.tail_mid_theta_5 = pyo.Constraint(m.N,
                                            rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, pyo_i(idx["theta_5"])],
                                                               (2 / 3) * np.pi))
        m.tail_mid_psi_5 = pyo.Constraint(m.N,
                                          rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, pyo_i(idx["psi_5"])],
                                                             (2 / 3) * np.pi))
        # Front left leg
        m.l_shoulder_theta_6 = pyo.Constraint(m.N,
                                              rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, pyo_i(idx["theta_6"])],
                                                                 (3 / 4) * np.pi))
        m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi, m.x[n, pyo_i(idx["theta_7"])], 0))
        # Front right leg
        m.r_shoulder_theta_8 = pyo.Constraint(m.N,
                                              rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, pyo_i(idx["theta_8"])],
                                                                 (3 / 4) * np.pi))
        m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi, m.x[n, pyo_i(idx["theta_9"])], 0))
        # Back left leg
        m.l_hip_theta_10 = pyo.Constraint(m.N,
                                          rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, pyo_i(idx["theta_10"])],
                                                             (3 / 4) * np.pi))
        m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=lambda m, n: (0, m.x[n, pyo_i(idx["theta_11"])], np.pi))
        # Back right leg
        m.r_hip_theta_12 = pyo.Constraint(m.N,
                                          rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, pyo_i(idx["theta_12"])],
                                                             (3 / 4) * np.pi))

        m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=lambda m, n: (0, m.x[n, pyo_i(idx["theta_13"])], np.pi))
        m.l_front_ankle_theta_14 = pyo.Constraint(m.N,
                                                  rule=lambda m, n: (-np.pi / 4, m.x[n, pyo_i(idx["theta_14"])],
                                                                     (3 / 4) * np.pi))
        m.r_front_ankle_theta_15 = pyo.Constraint(m.N,
                                                  rule=lambda m, n: (-np.pi / 4, m.x[n, pyo_i(idx["theta_15"])],
                                                                     (3 / 4) * np.pi))
        m.l_back_ankle_theta_16 = pyo.Constraint(m.N,
                                                 rule=lambda m, n:
                                                 (-(3 / 4) * np.pi, m.x[n, pyo_i(idx["theta_16"])], 0))
        m.r_back_ankle_theta_17 = pyo.Constraint(m.N,
                                                 rule=lambda m, n:
                                                 (-(3 / 4) * np.pi, m.x[n, pyo_i(idx["theta_17"])], 0))

    logger.info("Constraint initialisation...Done")

    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_pose_err = 0.0
        slack_motion_err = 0.0
        slack_meas_err = 0.0
        for n in m.N:
            # Model Error
            if use_gmm:
                x = [m.x[n, i] for i in range(1, P + 1)]
                if inc_obj_vel:
                    x += [m.dx[n, i] for i in range(1, root_dim + 1)]
                slack_pose_err += -pyomo_log(
                    sum([
                        w * norm_pdf_multivariate(np.asarray(x)[root_dim:], mu, cov)
                        for w, mu, cov in zip(pose_gmm.gmm.weights_, pose_gmm.gmm.means_, pose_gmm.gmm.covariances_)
                    ]))
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p]**2
                slack_motion_err += m.motion_err_weight[p] * m.slack_motion[n, p]**2
                if not reduced_space and not use_gmm:
                    slack_pose_err += m.pose_err_weight[p] * m.slack_pose[n, p]**2
            # Measurement Error
            for l in m.L:
                for d2 in m.D2:
                    for w in m.W:
                        slack_meas_err += loss_function(m.meas_err_weight[n, l, w] * m.slack_meas[n, l, d2, w], loss)
        return 1e-3 * (slack_meas_err + slack_model_err + slack_pose_err + slack_motion_err)

    m.obj = pyo.Objective(rule=obj)

    logger.info("Objective initialisation...Done")
    # RUN THE SOLVER
    if opt is None:
        opt = SolverFactory(
            "ipopt", executable="/home/zico/lib/ipopt/build/bin/ipopt" if platform.system() == "Linux" else None
        )
        # solver options
        opt.options["print_level"] = 5
        opt.options["max_iter"] = 1000
        opt.options["max_cpu_time"] = 10000
        opt.options["Tol"] = 1e-1
        opt.options["OF_print_timing_statistics"] = "yes"
        opt.options["OF_print_frequency_time"] = 10
        opt.options["OF_hessian_approximation"] = "limited-memory"
        opt.options["OF_accept_every_trial_step"] = "yes"
        opt.options["linear_solver"] = "ma86"
        opt.options["OF_ma86_scaling"] = "none"

    logger.info("Setup optimisation - End")
    t1 = time()
    logger.info(f"Initialisation took {t1 - t0:.2f}s")

    t0 = time()
    opt.solve(m, tee=True)
    t1 = time()
    logger.info(f"Optimisation solver took {t1 - t0:.2f}s")

    app.stop_logging()

    logger.info("Generate outputs...")

    # ===== SAVE FTE RESULTS =====
    if reduced_space:
        x_optimised = pose_model.project(get_vals_v(m.x, [m.N, m.P]), inverse=True)
    else:
        x_optimised = get_vals_v(m.x, [m.N, m.P])
    dx_optimised = get_vals_v(m.dx, [m.N, m.P])
    ddx_optimised = get_vals_v(m.ddx, [m.N, m.P])
    positions = [pose_to_3d(*states) for states in x_optimised]
    model_weight = get_vals_v(m.model_err_weight, [m.P])
    model_err = get_vals_v(m.slack_model, [m.N, m.P])
    pose_err = get_vals_v(m.slack_pose, [m.N, m.P])
    motion_err = get_vals_v(m.slack_motion, [m.N, m.P])
    meas_err = get_vals_v(m.slack_meas, [m.N, m.L, m.D2, m.W])
    meas_weight = get_vals_v(m.meas_err_weight, [m.N, m.L, m.W])

    states = dict(x=x_optimised,
                  dx=dx_optimised,
                  ddx=ddx_optimised,
                  model_err=model_err,
                  model_weight=model_weight,
                  meas_err=meas_err,
                  meas_weight=meas_weight,
                  pose_err=pose_err,
                  motion_err=motion_err)

    out_fpath = os.path.join(out_dir, "fte.pickle")
    app.save_optimised_cheetah(positions, out_fpath, extra_data=dict(**states, start_frame=start_frame))
    app.save_3d_cheetah_as_2d(positions,
                              out_dir,
                              scene_fpath,
                              markers,
                              project_points_fisheye,
                              start_frame,
                              out_fname="fte",
                              vid_dir=data_dir)

    # Create 2D reprojection videos.
    if generate_reprojection_videos:
        video_paths = sorted(glob(os.path.join(root_dir, "cheetah_videos", data_path,
                                               "cam[1-9].mp4")))  # original vids should be in the parent dir
        app.create_labeled_videos(video_paths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)

    # Calculate single view error.
    if extra_constraints:
        compare_traj_error(os.path.join(fte_dir, "fte.pickle"),
                           out_fpath,
                           os.path.join(root_dir, "cheetah_videos"),
                           data_path,
                           fte_type=fte_type,
                           out_dir_prefix=out_dir_prefix)

    logger.info("Done")


def dataset_post_process():
    import matplotlib.pyplot as plt
    root_dir = "/Users/zico/OneDrive - University of Cape Town/CheetahReconstructionResults/cheetah_videos"
    dir_prefix = "/Users/zico/msc/dev/AcinoSet/data"
    tests = {
        "2017_08_29/top/jules/run1_1": 1,
        "2017_09_02/top/jules/run1": 1,
        "2019_02_27/romeo/run": 3,
        "2017_09_03/top/zorro/run1_2": 4,
        "2017_08_29/top/phantom/run1_1": 1,
        "2017_09_02/top/phantom/run1_3": 4,
    }
    metrics = {
        "pixels": [],
        "full_single_traj_error": [],
        "single_traj_error": [],
        "single_pos_error": [],
        "model_pos_error": [],
        "distance": [],
        "angle": []
    }
    markers = misc.get_markers()

    def _pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
        y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
        z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
        #project onto camera plane
        a = x_2d / z_2d
        b = y_2d / z_2d
        #fisheye params
        r = (a**2 + b**2)**0.5
        th = np.arctan(r)
        #distortion
        th_d = th * (1 + D[0] * th**2 + D[1] * th**4 + D[2] * th**6 + D[3] * th**8)
        x_p = a * th_d / (r + 1e-12)
        y_p = b * th_d / (r + 1e-12)
        u = K[0, 0] * x_p + K[0, 2]
        v = K[1, 1] * y_p + K[1, 2]
        return u, v

    def _get_bounding_box(x, y, axis=1):
        x_range = np.max(x, axis=axis) - np.min(x, axis=axis)
        y_range = np.max(y, axis=axis) - np.min(y, axis=axis)
        return x_range, y_range

    points_dfs = []
    for i, data_path in enumerate(tests.keys()):
        cam_idx = tests[data_path]
        # Read the scene data file and the reconstruction data.
        k_arr, d_arr, r_arr, t_arr, _, n_cams, _ = utils.find_scene_file(os.path.join(root_dir, data_path))
        multi_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "sd_fte", "fte.pickle"))
        single_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte", "fte.pickle"))
        pose_model_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, f"fte_{cam_idx}", "fte.pickle"))

        # Read the camera 2D measurements.
        df_paths = sorted(glob(os.path.join(os.path.join(root_dir, data_path, "dlc"), "*.h5")))
        h5_filename = os.path.basename(df_paths[cam_idx])
        points_2d_df = pd.read_hdf(os.path.join(os.path.join(root_dir, data_path, "dlc", h5_filename)))
        # Re-organise data to make for easy selection of data in the dataframe.
        points_2d_df = points_2d_df.droplevel([0], axis=1).swaplevel(0, 1, axis=1).T.unstack().T.reset_index().rename(
            {"level_0": "frame"}, axis=1)
        points_2d_df.columns.name = ""
        points_2d_df.rename(columns={"bodyparts": "marker"}, inplace=True)
        points_2d_df = points_2d_df[points_2d_df.marker != "lure"]
        points_2d_df.loc[points_2d_df["likelihood"] <= 0.8, ["x", "y"]] = float("NaN")
        start_frame = multi_view_data['start_frame']
        end_frame = multi_view_data['start_frame'] + len(multi_view_data['positions'])
        points_2d = points_2d_df.query(f"{start_frame} <= frame < {end_frame}")
        points_2d = points_2d.assign(test=data_path)
        points_dfs.append(points_2d)

        # Calculate head distance from the camera.
        gt_pos = np.asarray(multi_view_data["positions"])
        for n in range(gt_pos.shape[0]):
            for m_idx, m in enumerate(markers):
                [u], [v] = _pt3d_to_2d(gt_pos[n, m_idx][0], gt_pos[n, m_idx][1], gt_pos[n, m_idx][2], k_arr[cam_idx],
                                       d_arr[cam_idx], r_arr[cam_idx], t_arr[cam_idx])
                metrics["pixels"].append({"test": data_path, "frame": start_frame + n, "marker": m, "x": u, "y": v})

        # Calculate the trajectory error per bodypart.
        single_mpjpe_mm, single_traj_error = traj_error(multi_view_data["positions"], single_view_data["positions"])
        _, full_single_mpjpe_mm = traj_error(multi_view_data["positions"],
                                             single_view_data["positions"],
                                             centered=False)
        model_mpjpe_mm, _ = traj_error(multi_view_data["positions"], pose_model_data["positions"])
        metrics["full_single_traj_error"].append(full_single_mpjpe_mm.tolist())
        metrics["single_traj_error"].append(single_traj_error.tolist())
        metrics["single_pos_error"].append(single_mpjpe_mm.transpose())
        metrics["model_pos_error"].append(model_mpjpe_mm.transpose())

    single_total_error = pd.concat(metrics["single_pos_error"]).mean()
    model_total_error = pd.concat(metrics["model_pos_error"]).mean()
    df = pd.concat([single_total_error, model_total_error], axis=1)
    df.columns = [
        f"single-view ({single_total_error.mean():.2f})", f"single-view with model ({model_total_error.mean():.2f})"
    ]

    fig = plt.figure(figsize=(16, 12), dpi=120)
    full_single_mpjpe_mm = np.array(sum(metrics["full_single_traj_error"], []))
    temp = pd.DataFrame.from_dict(metrics["pixels"])
    temp2 = pd.concat(points_dfs, ignore_index=True)

    error_dfs = []
    for data_path in list(tests.keys()):
        df1 = temp.query(f"test == '{data_path}'")
        df2 = temp2.query(f"test == '{data_path}'")
        valid_frames = np.intersect1d(df1['frame'].to_numpy(), df2['frame'].to_numpy())
        for m in markers:
            # extract frames
            q = f'marker == "{m}"'
            pts_1_df = df1.query(q)
            pts_2_df = df2.query(q)
            pts_1_df = pts_1_df[pts_1_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            pts_2_df = pts_2_df[pts_2_df['frame'].isin(valid_frames)].sort_values(by=['frame'])

            # get 2d and reprojected points
            pts_1 = pts_1_df[['x', 'y']].to_numpy()
            pts_2 = pts_2_df[['x', 'y']].to_numpy()

            # compare both types of points
            residual = np.sqrt(np.sum((pts_1 - pts_2)**2, axis=1))
            # make the result dataframe
            marker_arr = np.array([m] * len(valid_frames))
            error_dfs.append(
                pd.DataFrame(np.vstack((valid_frames, marker_arr, residual)).T,
                             columns=['frame', 'marker', 'pixel_residual']))

    result_df = pd.concat(error_dfs, ignore_index=True).sort_values(
        by=['frame']) if len(error_dfs) > 0 else pd.DataFrame(
            columns=['frame', 'marker', 'pixel_residual']).sort_values(by=['frame'])
    cam_px = temp[["x", "y"]].to_numpy(dtype=float).reshape((-1, len(markers), 2))
    x_bounds, y_bounds = _get_bounding_box(cam_px[:, :, 0], cam_px[:, :, 1])
    bounding_boxes = np.sqrt(x_bounds * y_bounds)
    gt_2d = result_df["pixel_residual"].to_numpy(dtype=float).reshape((-1, len(markers)))
    mean_gt_2d = np.nanmean(gt_2d, axis=1)
    num_nans = np.count_nonzero(np.isnan(gt_2d), axis=1)
    ax = fig.add_subplot(2, 2, (1, 3))
    ax.scatter(bounding_boxes, full_single_mpjpe_mm)
    ax.set_xlabel("Bounding Box Size")
    ax.set_ylabel("Avg. 3D Error (mm)")
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(bounding_boxes, mean_gt_2d)
    ax.set_xlabel("Bounding Box Size")
    ax.set_ylabel("Avg. Reprojection Error (px)")
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(bounding_boxes, num_nans)
    ax.set_xlabel("Bounding Box Size")
    ax.set_ylabel("# of Missing Measurements")
    fig.savefig("../data/bounding_box_vs_error.png")
    plt.cla()

    ax = df.plot(kind="barh")
    fig = ax.get_figure()
    plt.xlabel("Error (mm)")
    plt.ylabel("Bodypart")
    fig.savefig("../data/mpjpe_result.png")
    plt.cla()


def run_dataset_test():
    import gc
    working_dir = "/Users/zico/OneDrive - University of Cape Town/CheetahReconstructionResults/cheetah_videos"
    dir_prefix = "/Users/zico/msc/dev/AcinoSet/data"
    tests = {
        "2017_08_29/top/jules/run1_1": (10, 110, 1),
        "2017_09_02/top/phantom/run1_3": (50, 130, 4),
        "2017_09_02/top/jules/run1": (21, 100, 1),
        "2017_08_29/top/phantom/run1_1": (20, 170, 1),
        "2019_02_27/romeo/run": (100, 190, 3),
        "2017_09_03/top/zorro/run1_2": (20, 120, 4)
    }
    time0 = time()
    logger.info("Run through subset...")
    # Initialise the Ipopt solver.
    optimiser = SolverFactory("ipopt")  #, executable="/home/zico/lib/ipopt/build/bin/ipopt")
    # solver options
    optimiser.options["print_level"] = 5
    optimiser.options["max_iter"] = 500
    optimiser.options["max_cpu_time"] = 10000
    optimiser.options["Tol"] = 1e-1
    optimiser.options["OF_print_timing_statistics"] = "yes"
    optimiser.options["OF_print_frequency_time"] = 10
    optimiser.options["OF_hessian_approximation"] = "limited-memory"
    optimiser.options["OF_accept_every_trial_step"] = "yes"
    optimiser.options["linear_solver"] = "ma86"
    optimiser.options["OF_ma86_scaling"] = "none"
    for test_vid in tqdm(tests.keys()):
        # Force garbage collection so that the repeated model creation does not overflow the memory!
        gc.collect()
        run(working_dir,
            test_vid,
            cam_idx=tests[test_vid][2],
            start_frame=tests[test_vid][0],
            end_frame=tests[test_vid][1],
            dlc_thresh=0.5,
            opt=optimiser,
            out_dir_prefix=dir_prefix,
            generate_reprojection_videos=False)

    time1 = time()
    logger.info(f"Run through all videos took {time1 - time0:.2f}s")


def run_tests():
    import gc
    working_dir = os.path.join('/', 'data', 'dlc', 'to_analyse', 'cheetah_videos')
    dir_prefix = "/data/zico/CheetahResults/pw_sd_fte"
    model_dir = os.path.join("/data/zico/CheetahRuns/v4")
    straight_runs = [
        "2017_08_29/top/jules/run1_1",
        "2017_09_02/top/phantom/run1_1",
        "2017_09_02/top/phantom/run1_3",
        "2017_09_02/top/jules/run1",
        "2017_09_02/bottom/jules/run2",
        "2017_09_03/bottom/zorro/run2_2",
        "2017_09_03/top/zorro/run1_2",
        "2017_12_16/top/cetane/run1",
        "2017_12_21/top/lily/run1",
        # End of dataset used for reduced pose representation with PCA.
        "2019_02_27/ebony/run",
        "2019_02_27/kiara/run",
        "2019_02_27/romeo/run",
        "2019_03_03/menya/run",
        "2019_03_05/jules/run",
        "2017_08_29/top/phantom/run1_1",
        "2017_08_29/top/jules/run1_2",
        "2017_09_03/bottom/zorro/run2_1",
        "2017_12_10/top/phantom/run1",
        "2017_12_12/top/cetane/run1_2",
        "2017_12_12/top/cetane/run1_1",
        "2019_03_05/lily/run",
        "2019_03_07/menya/run",
        "2019_03_07/phantom/run"
    ]
    check_straight_runs = [run.replace("/", "_") for run in straight_runs]
    # Read in all pickle files to gather trajectories into a dictionary of trajectories.
    test_files = glob(os.path.join(model_dir, "*.pickle"))
    tests = {}
    for t_fname in test_files:
        fname = t_fname.split(model_dir + "/")[-1]
        input_data = data_ops.load_pickle(t_fname)
        test_key = fname.split(".pickle")[0]
        if test_key in check_straight_runs:
            test = straight_runs[check_straight_runs.index(test_key)]
            tests[test] = {}
            tests[test]["start_frame"] = input_data["start_frame"] + 1
            tests[test]["end_frame"] = input_data["start_frame"] + np.asarray(input_data["positions"]).shape[0]

    logger.info(f"Number of points in each run trajectory ({len(tests.keys())}):")
    [print(f"{t_run} - ({tests[t_run]['start_frame']}, {tests[t_run]['end_frame']})") for t_run in tests.keys()]

    time0 = time()
    logger.info("Run through subset...")
    # Initialise the Ipopt solver.
    optimiser = SolverFactory("ipopt", executable="/home/zico/lib/ipopt/build/bin/ipopt")
    # solver options
    optimiser.options["print_level"] = 5
    optimiser.options["max_iter"] = 500
    optimiser.options["max_cpu_time"] = 10000
    optimiser.options["Tol"] = 1e-1
    optimiser.options["OF_print_timing_statistics"] = "yes"
    optimiser.options["OF_print_frequency_time"] = 10
    optimiser.options["OF_hessian_approximation"] = "limited-memory"
    optimiser.options["OF_accept_every_trial_step"] = "yes"
    optimiser.options["linear_solver"] = "ma86"
    optimiser.options["OF_ma86_scaling"] = "none"
    for test_vid in tqdm(tests.keys()):
        for i in range(6):
            # Force garbage collection so that the repeated model creation does not overflow the memory!
            gc.collect()
            try:
                run(working_dir,
                    test_vid,
                    cam_idx=i,
                    start_frame=tests[test_vid]["start_frame"],
                    end_frame=tests[test_vid]["end_frame"],
                    dlc_thresh=0.5,
                    n_comps=9,
                    opt=optimiser,
                    out_dir_prefix=dir_prefix,
                    init_fte=True,
                    extra_constraints=False,
                    generate_reprojection_videos=False)
            except Exception:
                continue

    time1 = time()
    logger.info(f"Run through all videos took {time1 - time0:.2f}s")


if __name__ == "__main__":
    # run_dataset_test()
    # dataset_post_process()
    run_tests()
