import os
import platform
import json
from typing import Union, Tuple
import numpy as np
from pyomo.core.expr.current import log as pyomo_log
import sympy as sp
import pandas as pd
from glob import glob
from time import time
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from lib import misc, utils, app
from lib.calib import triangulate_points_fisheye, project_points_fisheye
from py_utils import data_ops, log
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Create a module logger with the name of this file.
logger = log.logger(__name__)


def pose_distance(X: list, Y: list):
    distance = [pyo.sqrt(sum((a - b)**2 for a, b in zip(X_row, Y[i]))) for i, X_row in enumerate(X)]
    return sum(distance) / len(distance)


def mat_mul(X: list, Y: list):
    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)] for X_row in X]


def vec_add(x: list, y: list):
    return [a + b for a, b in zip(x, y)]


def vec_sub(x: list, y: list):
    return [a - b for a, b in zip(x, y)]


def vec_scale(x: list, y: list):
    return [a * b for a, b in zip(x, y)]


def scale_motion(X: list, scale: list, translate: list) -> list:
    # X *= self.scale_ X += self.min_
    X_ = sum(X, [])
    X_temp = [a * b for a, b in zip(X_, scale)]
    return [a + b for a, b in zip(X_temp, translate)]


def predict_motion(X: list, B: list, C: list, D: list = None, E: list = None) -> list:
    if D and E: [y_pred] = mat_mul([scale_motion(X, D, E)], B)
    else: [y_pred] = mat_mul(X, B)

    return [a + b for a, b in zip(y_pred, C)]


def train_motion_model(n: int, start_idx: int, window_size: int, window_time: int) -> Tuple[Pipeline, np.ndarray]:
    logger.info("Train motion prediction model")
    model_dir = "/Users/zico/msc/data/CheetahRuns/v4/model"
    df = pd.read_hdf(os.path.join(model_dir, "dataset_runs.h5"))
    idx = np.where(df.index.values == 0)[0]
    df_in = df.iloc[:, start_idx:start_idx + n]
    df_list = []
    end_segment = 0
    for begin_segment, end_segment in zip(idx, idx[1:]):
        df_list.append(
            data_ops.series_to_supervised(df_in.iloc[begin_segment:end_segment], n_in=window_size, n_step=window_time))
    df_list.append(data_ops.series_to_supervised(df_in.iloc[end_segment:], n_in=window_size, n_step=window_time))
    df_input = pd.concat(df_list)

    pipeline = Pipeline(steps=[("norm", preprocessing.MinMaxScaler()), ("model", LinearRegression())])
    xy_set = df_input.to_numpy()
    X = xy_set[:, 0:(n * window_size)]
    y = xy_set[:, (n * window_size):]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    pipeline.fit(X_train, y_train)
    B = pipeline["model"].coef_.T.tolist()
    C = pipeline["model"].intercept_.tolist()
    D = pipeline["norm"].scale_.tolist()
    E = pipeline["norm"].min_.tolist()
    y_pred = pipeline.predict(X_test)

    y_pred_manual = np.asarray([predict_motion([pose.tolist()], B, C, D, E) for pose in X_test])
    check_motion_predict = (y_pred - y_pred_manual)**2
    if np.sum(check_motion_predict) < 1e-15:
        logger.info("Pure Python LR prediction matches sklearn :)")

    residuals = y_test - y_pred_manual
    variance = np.var(residuals, axis=0)
    logger.info(f"Model prediction error: {np.mean(residuals**2)}")

    return pipeline, np.asarray(variance)


def traj_error(X: np.ndarray, Y: np.ndarray, plot: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    markers = misc.get_markers()
    X -= np.expand_dims(np.mean(X, axis=1), axis=1)
    Y -= np.expand_dims(np.mean(Y, axis=1), axis=1)
    distances = np.sqrt(np.sum((X - Y)**2, axis=2))
    trajectory_error_mm = np.mean(distances, axis=1) * 1000.0
    mpjpe_mm = np.mean(distances, axis=0) * 1000.0
    result = pd.DataFrame(mpjpe_mm.reshape(1, len(markers)), columns=markers)
    logger.info(f"Single view error [mm] across the entire trajectory: {float(result.mean(axis=1))}")

    return result.transpose(), trajectory_error_mm


def compare_traj_error(fte_orig: str,
                       fte: str,
                       root_dir: str,
                       data_dir: str,
                       fte_type: str = "sd_fte",
                       out_dir_prefix: str = None) -> None:
    import matplotlib.pyplot as plt

    if out_dir_prefix:
        fte_multi_view = os.path.join(out_dir_prefix, data_dir, fte_type, "fte.pickle")
    else:
        fte_multi_view = os.path.join(root_dir, data_dir, fte_type, "fte.pickle")
    multi_view_data = data_ops.load_pickle(fte_multi_view)
    single_view_data = data_ops.load_pickle(fte_orig)
    pose_model_data = data_ops.load_pickle(fte)
    _, single_view_error = traj_error(np.asarray(multi_view_data["positions"]),
                                      np.asarray(single_view_data["positions"]))
    _, pose_model_error = traj_error(np.asarray(multi_view_data["positions"]), np.asarray(pose_model_data["positions"]))
    plt.figure()
    plt.plot(single_view_error, label="Single View")
    plt.plot([np.mean(single_view_error)] * len(single_view_error), label="Mean Single View", linestyle="--")
    plt.plot(pose_model_error, label="Single View Motion Prior")
    plt.plot([np.mean(pose_model_error)] * len(pose_model_error), label="Mean Single View Motion Prior", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Error (mm)")
    ax = plt.gca()
    ax.legend()
    plt.show(block=True)


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
        loss="redescending",
        n_comps: int = 5,
        pairwise_included: int = 0,
        reduced_space: bool = False,
        inc_obj_orien: bool = False,
        inc_obj_vel: bool = True,
        init_fte=False,
        opt=None,
        out_dir_prefix: str = None,
        generate_reprojection_videos: bool = False):
    logger.info("Prepare data - Start")

    t0 = time()

    if out_dir_prefix:
        out_dir = os.path.join(out_dir_prefix, data_path, f"fte_{cam_idx}")
    else:
        out_dir = os.path.join(root_dir, data_path, f"fte_{cam_idx}")

    data_dir = os.path.join(root_dir, data_path)
    assert os.path.exists(data_dir)
    dlc_dir = os.path.join(data_dir, "dlc_pw")
    assert os.path.exists(dlc_dir)
    os.makedirs(out_dir, exist_ok=True)

    app.start_logging(os.path.join(out_dir, "fte.log"))

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

    # load measurement dataframe (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
    filtered_points_2d_df = points_2d_df[points_2d_df["likelihood"] > dlc_thresh]  # ignore points with low likelihood

    if platform.python_implementation() == "PyPy":
        # At the moment video reading does not work with openCV and PyPy - well at least not on the Linux i9.
        # So instead I manually get the number of frames and the frame rate based (determined above).
        num_frames = points_2d_df["frame"].max() + 1

    assert 0 != start_frame < num_frames, f"start_frame must be strictly between 0 and {num_frames}"
    assert 0 != end_frame <= num_frames, f"end_frame must be less than or equal to {num_frames}"
    assert 0 <= dlc_thresh <= 1, "dlc_thresh must be from 0 to 1"

    if end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        target_markers = misc.get_markers()
        markers_condition = " or ".join([f"marker=='{ref}'" for ref in target_markers])
        num_marker = lambda i: len(
            filtered_points_2d_df.query(f"frame == {i} and ({markers_condition})")["marker"].unique())

        start_frame, end_frame = -1, -1
        max_idx = points_2d_df["frame"].max() + 1
        for i in range(max_idx):
            if num_marker(i) == len(target_markers):
                start_frame = i
                break
        for i in range(max_idx, 0, -1):
            if num_marker(i) == len(target_markers):
                end_frame = i
                break
        if start_frame == -1 or end_frame == -1:
            raise Exception("Setting frames failed. Please define start and end frames manually.")
    elif start_frame == -1:
        # Use the entire video.
        start_frame = 1
        end_frame = num_frames
    else:
        # User-defined frames
        start_frame = start_frame - 1  # 0 based indexing
        end_frame = end_frame % num_frames + 1 if end_frame == -1 else end_frame
    assert len(k_arr) == points_2d_df["camera"].nunique()

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
    pose_to_3d, pos_funcs = data_ops.load_dill(os.path.join(root_dir, "pose_3d_functions_with_paws.pickle"))
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
    #===================================================
    #                   Load in data
    #===================================================
    logger.info("Load H5 2D DLC prediction data")
    df_paths = sorted(glob(os.path.join(dlc_dir, "*.h5")))

    points_3d_df = utils.get_pairwise_3d_points_from_df(filtered_points_2d_df, k_arr, d_arr, r_arr, t_arr,
                                                        triangulate_points_fisheye)

    # estimate initial points
    logger.info("Estimate the initial trajectory")
    # Use the cheetahs spine to estimate the initial trajectory with a 3rd degree spline.
    frame_est = np.arange(end_frame)

    nose_pts = points_3d_df[points_3d_df["marker"] == "nose"][["frame", "x", "y", "z"]].values
    nose_pts[:, 1] = nose_pts[:, 1] - 0.055
    nose_pts[:, 3] = nose_pts[:, 3] + 0.055
    traj_est_x = UnivariateSpline(nose_pts[:, 0], nose_pts[:, 1])
    traj_est_y = UnivariateSpline(nose_pts[:, 0], nose_pts[:, 2])
    traj_est_z = UnivariateSpline(nose_pts[:, 0], nose_pts[:, 3])
    x_est = np.array(traj_est_x(frame_est))
    y_est = np.array(traj_est_y(frame_est))
    z_est = np.array(traj_est_z(frame_est))

    # Calculate the initial yaw.
    dx_est = np.diff(x_est) / Ts
    dy_est = np.diff(y_est) / Ts
    psi_est = np.arctan2(dy_est, dx_est)
    # Duplicate the last heading estimate as the difference calculation returns N-1.
    psi_est = np.append(psi_est, [psi_est[-1]])

    # Remove datafames from memory to conserve memory usage.
    del points_2d_df
    del filtered_points_2d_df
    del points_3d_df

    # Pairwise correspondence data.
    h5_filename = os.path.basename(df_paths[cam_idx])
    pw_data = data_ops.load_pickle(
        os.path.join(dlc_dir, f"{h5_filename[:4]}DLC_resnet152_CheetahOct14shuffle4_650000.pickle"))
    base_data = list(pd.read_hdf(os.path.join(dlc_dir, h5_filename)).to_numpy())

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
    pose_model = misc.PoseModel("/Users/zico/msc/data/CheetahRuns/v4/model/dataset_pose.h5",
                                pose_params=idx,
                                ext_dim=ext_dim,
                                n_comps=n_comps,
                                standardise=True)
    pose_var = 0.3 * pose_model.error_variance
    if reduced_space:
        Q = np.abs(pose_model.project(Q))
    # ======= WEIGHTS =======
    def init_meas_weights(m, n, l, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        marker = markers[l - 1]
        values = pw_data[(n - 1) + start_frame]
        likelihoods = values['pose'][2::3]
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
                val = values['pose'][d2 - 1::3]
                base = pair_dict[marker][w - 2]
                val_pw = values['pws'][:, :, :, d2 - 1]
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

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
    if init_fte:
        fte_states = data_ops.load_pickle(os.path.join(os.path.dirname(out_dir), "sd_fte", "fte.pickle"))
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
            var_list = pose_model.project(np.asarray(var_list), inverse=True)
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
            var_list = pose_model.project(np.asarray(var_list), inverse=True)
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

    if not reduced_space:

        def reduced_pose_constraint(m, n, p):
            # Contrain poses that are close to the reduced pose space.
            x = [m.x[n, i] for i in range(1, P + 1)]
            if inc_obj_vel:
                x += [m.dx[n, i] for i in range(1, root_dim + 1)]
            x_r = pose_model.project(pose_model.project(np.asarray(x)), inverse=True).tolist()
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

    logger.info("Constaint initialisation...Done")

    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_pose_err = 0.0
        slack_meas_err = 0.0
        for n in m.N:
            # Model Error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p]**2
                if not reduced_space:
                    slack_pose_err += m.pose_err_weight[p] * m.slack_pose[n, p]**2
            # Measurement Error
            for l in m.L:
                for d2 in m.D2:
                    for w in m.W:
                        slack_meas_err += loss_function(m.meas_err_weight[n, l, w] * m.slack_meas[n, l, d2, w], loss)
        return 1e-3 * (slack_meas_err + slack_model_err + slack_pose_err)

    m.obj = pyo.Objective(rule=obj)

    logger.info("Objective initialisation...Done")
    # RUN THE SOLVER
    if opt is None:
        opt = SolverFactory(
            "ipopt",  #executable="/home/zico/lib/ipopt/build/bin/ipopt"
        )
        # solver options
        opt.options["print_level"] = 5
        opt.options["max_iter"] = 400
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
    x_optimised = get_vals_v(m.x, [m.N, m.P])
    dx_optimised = get_vals_v(m.dx, [m.N, m.P])
    ddx_optimised = get_vals_v(m.ddx, [m.N, m.P])
    if reduced_space:
        positions = [pose_to_3d(*states) for states in pose_model.project(x_optimised, inverse=True)]
    else:
        positions = [pose_to_3d(*states) for states in x_optimised]
    model_weight = get_vals_v(m.model_err_weight, [m.P])
    model_err = get_vals_v(m.slack_model, [m.N, m.P])
    pose_err = get_vals_v(m.slack_pose, [m.N, m.P])
    meas_err = get_vals_v(m.slack_meas, [m.N, m.L, m.D2, m.W])
    meas_weight = get_vals_v(m.meas_err_weight, [m.N, m.L, m.W])

    states = dict(x=x_optimised,
                  dx=dx_optimised,
                  ddx=ddx_optimised,
                  model_err=model_err,
                  model_weight=model_weight,
                  meas_err=meas_err,
                  meas_weight=meas_weight,
                  motion_err=pose_err)

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
        video_paths = sorted(glob(os.path.join(root_dir, data_path,
                                               "cam[1-9].mp4")))  # original vids should be in the parent dir
        app.create_labeled_videos(video_paths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)

    # Calculate single view error.
    multi_view_data = data_ops.load_pickle(os.path.join(os.path.dirname(out_dir), "sd_fte", "fte.pickle"))
    results = traj_error(np.asarray(multi_view_data["positions"]), np.asarray(positions))

    logger.info("Done")


if __name__ == "__main__":
    import gc
    working_dir = os.path.join("/", "data", "dlc", "to_analyse", "cheetah_videos")
    video_data = data_ops.load_pickle("/data/zico/CheetahResults/test_videos_list.pickle")
    tests = video_data["test_dirs"]
    dir_prefix = "/data/zico/CheetahResults/paws-included"
    manually_selected_frames = {
        "2019_03_03/phantom/run": (73, 272),
        "2017_12_12/top/cetane/run1_1": (100, 241),
        "2019_03_05/jules/run": (58, 176),
        "2019_03_09/lily/run": (80, 180),
        "2017_09_03/top/zorro/run1_2": (20, 120),
        "2017_08_29/top/phantom/run1_1": (20, 170),
        "2017_12_21/top/lily/run1": (7, 106),
        "2019_03_03/menya/run": (20, 130),
        "2017_12_10/top/phantom/run1": (30, 130),
        "2017_09_03/bottom/zorro/run2_1": (126, 325),
        "2019_02_27/ebony/run": (20, 200),
        "2017_12_09/bottom/phantom/run2": (18, 117),
        "2017_09_03/bottom/zorro/run2_3": (1, 200),
        "2017_08_29/top/jules/run1_1": (10, 110),
        "2017_09_02/top/jules/run1": (10, 110),
        "2019_03_07/menya/run": (60, 160),
        "2017_09_02/top/phantom/run1_2": (20, 160),
        "2019_03_05/lily/run": (150, 250),
        "2017_12_12/top/cetane/run1_2": (3, 202),
        "2019_03_07/phantom/run": (100, 200),
        "2019_02_27/romeo/run": (12, 190),
        "2017_08_29/top/jules/run1_2": (30, 130),
        "2017_12_16/top/cetane/run1": (110, 210),
        "2017_09_02/top/phantom/run1_1": (33, 150),
        "2017_09_02/top/phantom/run1_3": (35, 135),
        "2017_09_03/top/zorro/run1_1": (4, 203),
        "2019_02_27/kiara/run": (10, 110),
        "2017_09_02/bottom/jules/run2": (35, 171),
        "2017_09_03/bottom/zorro/run2_2": (32, 141)
    }
    bad_videos = ("2017_09_03/bottom/phantom/flick2", "2017_09_02/top/phantom/flick1_1", "2017_12_17/top/zorro/flick1")
    if platform.python_implementation() == "PyPy":
        time0 = time()
        logger.info("Run reconstruction on all videos...")
        # Initialise the Ipopt solver.
        optimiser = SolverFactory("ipopt", executable="/home/zico/lib/ipopt/build/bin/ipopt")
        # solver options
        optimiser.options["print_level"] = 5
        optimiser.options["max_iter"] = 400
        optimiser.options["max_cpu_time"] = 10000
        optimiser.options["Tol"] = 1e-1
        optimiser.options["OF_print_timing_statistics"] = "yes"
        optimiser.options["OF_print_frequency_time"] = 10
        optimiser.options["OF_hessian_approximation"] = "limited-memory"
        optimiser.options["OF_accept_every_trial_step"] = "yes"
        optimiser.options["linear_solver"] = "ma86"
        optimiser.options["OF_ma86_scaling"] = "none"
        valid_vids = set(manually_selected_frames.keys())
        for test_vid in tqdm(tests):
            # Force garbage collection so that the repeated model creation does not overflow the memory!
            gc.collect()
            current_dir = test_vid.split("/cheetah_videos/")[1]
            # Filter parameters based on past experience.
            if current_dir not in valid_vids:
                # Skip these videos because of erroneous input data.
                continue
            start = 1
            end = -1
            if current_dir in set(manually_selected_frames.keys()):
                start = manually_selected_frames[current_dir][0]
                end = manually_selected_frames[current_dir][1]
            try:
                run(working_dir,
                    current_dir,
                    start_frame=start,
                    end_frame=end,
                    dlc_thresh=0.5,
                    opt=optimiser,
                    out_dir_prefix=dir_prefix)
            except Exception:
                run(working_dir,
                    current_dir,
                    start_frame=-1,
                    end_frame=1,
                    dlc_thresh=0.5,
                    opt=optimiser,
                    out_dir_prefix=dir_prefix)

        time1 = time()
        logger.info(f"Run through all videos took {time1 - time0:.2f}s")
    elif platform.python_implementation() == "CPython":
        time0 = time()
        logger.info("Run 2D reprojections on all videos...")
        for test_vid in tqdm(tests):
            current_dir = test_vid.split("/cheetah_videos/")[1]
            # Filter parameters based on past experience.
            if current_dir in bad_videos:
                # Skip these videos because of erroneous input data.
                continue
            try:
                if dir_prefix:
                    out_directory = os.path.join(dir_prefix, current_dir, "fte_pw")
                else:
                    out_directory = os.path.join(working_dir, current_dir, "fte_pw")
                video_fpaths = sorted(glob(os.path.join(working_dir, current_dir,
                                                        "cam[1-9].mp4")))  # original vids should be in the parent dir
                app.create_labeled_videos(video_fpaths, out_dir=out_directory, draw_skeleton=True, pcutoff=0.5)
            except Exception:
                continue

        time1 = time()
        logger.info(f"Video generation took {time1 - time0:.2f}s")
