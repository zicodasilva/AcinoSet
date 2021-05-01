import os
import numpy as np
import sympy as sp
import pandas as pd
from glob import glob
from time import time
from scipy.stats import linregress
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from lib import misc, utils, app
from lib.calib import triangulate_points_fisheye, project_points_fisheye
from py_utils import data_ops, log

# Create a module logger with the name of this file.
logger = log.logger(__name__)

def compare(first_file: str, second_file: str):
    data_fpaths = [first_file, second_file]
    app.plot_multiple_cheetah_reconstructions(data_fpaths, reprojections=False, dark_mode=True)

def display_test_image(data_dir, cam_num, pw_values, frame_num):
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    markers = misc.get_markers()
    marker_colors = cm.rainbow(np.linspace(0, 1, len(markers)))
    frame = os.path.join(data_dir, "frames", f"cam{cam_num}", f"frame{frame_num}.png")
    for pw in pw_values:
        # Plot the image.
        image = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        plt.figure(pw)
        plt.imshow(image)
        df = pd.read_hdf(os.path.join(data_dir, "fte_pw", "measurements", f"cam{cam_num}_pw_{pw}.h5"))
        for idx, marker in enumerate(markers):
            plt.plot(df.loc[frame_num-1][marker]["x"],
                        df.loc[frame_num-1][marker]["y"],
                        ".",
                        markersize=10,
                        color=marker_colors[idx],
                        label=marker)
            plt.legend(loc="upper right", fontsize="xx-small")
    # Show the plot.
    plt.show()

def create_pose_functions(data_dir):
     ## ========= POSE FUNCTIONS ========
    #SYMBOLIC ROTATION MATRIX FUNCTIONS
    def rot_x(x):
        c = sp.cos(x)
        s = sp.sin(x)
        return sp.Matrix([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])

    def rot_y(y):
        c = sp.cos(y)
        s = sp.sin(y)
        return sp.Matrix([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])

    def rot_z(z):
        c = sp.cos(z)
        s = sp.sin(z)
        return sp.Matrix([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

    L = 14  # number of joints in the cheetah model

    # defines arrays of angles, velocities and accelerations
    phi     = [sp.symbols(f"\\phi_{{{l}}}")   for l in range(L)]
    theta   = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
    psi     = [sp.symbols(f"\\psi_{{{l}}}")   for l in range(L)]

    #ROTATIONS
    RI_0 = rot_z(psi[0]) @ rot_x(phi[0]) @ rot_y(theta[0]) # head
    R0_I = RI_0.T
    RI_1 = rot_z(psi[1]) @ rot_x(phi[1]) @ rot_y(theta[1]) @ RI_0 # neck
    R1_I = RI_1.T
    RI_2 = rot_y(theta[2]) @ RI_1 # front torso
    R2_I = RI_2.T
    RI_3 = rot_z(psi[3])@ rot_x(phi[3]) @ rot_y(theta[3]) @ RI_2 # back torso
    R3_I = RI_3.T
    RI_4 = rot_z(psi[4]) @ rot_y(theta[4]) @ RI_3 # tail base
    R4_I = RI_4.T
    RI_5 = rot_z(psi[5]) @ rot_y(theta[5]) @ RI_4 # tail mid
    R5_I = RI_5.T
    RI_6 = rot_y(theta[6]) @ RI_2 # l_shoulder
    R6_I = RI_6.T
    RI_7 = rot_y(theta[7]) @ RI_6 # l_front_knee
    R7_I = RI_7.T
    RI_8 = rot_y(theta[8]) @ RI_2 # r_shoulder
    R8_I = RI_8.T
    RI_9 = rot_y(theta[9]) @ RI_8 # r_front_knee
    R9_I = RI_9.T
    RI_10 = rot_y(theta[10]) @ RI_3 # l_hip
    R10_I = RI_10.T
    RI_11 = rot_y(theta[11]) @ RI_10 # l_back_knee
    R11_I = RI_11.T
    RI_12 = rot_y(theta[12]) @ RI_3 # r_hip
    R12_I = RI_12.T
    RI_13 = rot_y(theta[13]) @ RI_12 # r_back_knee
    R13_I = RI_13.T

    # defines the position, velocities and accelerations in the inertial frame
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")
    # x_l, y_l, z_l = sp.symbols("x_l y_l z_l") # exclude lure for now


    # SYMBOLIC CHEETAH POSE POSITIONS
    p_head          = sp.Matrix([x, y, z])

    p_l_eye         = p_head         + R0_I  @ sp.Matrix([0, 0.03, 0])
    p_r_eye         = p_head         + R0_I  @ sp.Matrix([0, -0.03, 0])
    p_nose          = p_head         + R0_I  @ sp.Matrix([0.055, 0, -0.055])

    p_neck_base     = p_head         + R1_I  @ sp.Matrix([-0.28, 0, 0])
    p_spine         = p_neck_base    + R2_I  @ sp.Matrix([-0.37, 0, 0])

    p_tail_base     = p_spine        + R3_I  @ sp.Matrix([-0.37, 0, 0])
    p_tail_mid      = p_tail_base    + R4_I  @ sp.Matrix([-0.28, 0, 0])
    p_tail_tip      = p_tail_mid     + R5_I  @ sp.Matrix([-0.36, 0, 0])

    p_l_shoulder    = p_neck_base    + R2_I  @ sp.Matrix([-0.04, 0.08, -0.10])
    p_l_front_knee  = p_l_shoulder   + R6_I  @ sp.Matrix([0, 0, -0.24])
    p_l_front_ankle = p_l_front_knee + R7_I  @ sp.Matrix([0, 0, -0.28])

    p_r_shoulder    = p_neck_base    + R2_I  @ sp.Matrix([-0.04, -0.08, -0.10])
    p_r_front_knee  = p_r_shoulder   + R8_I  @ sp.Matrix([0, 0, -0.24])
    p_r_front_ankle = p_r_front_knee + R9_I  @ sp.Matrix([0, 0, -0.28])

    p_l_hip         = p_tail_base    + R3_I  @ sp.Matrix([0.12, 0.08, -0.06])
    p_l_back_knee   = p_l_hip        + R10_I @ sp.Matrix([0, 0, -0.32])
    p_l_back_ankle  = p_l_back_knee  + R11_I @ sp.Matrix([0, 0, -0.25])

    p_r_hip         = p_tail_base    + R3_I  @ sp.Matrix([0.12, -0.08, -0.06])
    p_r_back_knee   = p_r_hip        + R12_I @ sp.Matrix([0, 0, -0.32])
    p_r_back_ankle  = p_r_back_knee  + R13_I @ sp.Matrix([0, 0, -0.25])

    # p_lure          = sp.Matrix([x_l, y_l, z_l])

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    positions = sp.Matrix([
        p_l_eye.T, p_r_eye.T, p_nose.T,
        p_neck_base.T, p_spine.T,
        p_tail_base.T, p_tail_mid.T, p_tail_tip.T,
        p_l_shoulder.T, p_l_front_knee.T, p_l_front_ankle.T,
        p_r_shoulder.T, p_r_front_knee.T, p_r_front_ankle.T,
        p_l_hip.T, p_l_back_knee.T, p_l_back_ankle.T,
        p_r_hip.T, p_r_back_knee.T, p_r_back_ankle.T,
    #     p_lure.T
    ])

    func_map = {"sin": pyo.sin, "cos": pyo.cos, "ImmutableDenseMatrix":np.array}
    sym_list = [x, y, z,
                *phi, *theta, *psi,
    #             x_l, y_l, z_l
               ]
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    # Save the functions to file.
    data_ops.save_sympy_functions(os.path.join(data_dir, "pose_3d_functions.pickle"), (pose_to_3d, pos_funcs))

def run(root_dir: str, data_path: str, start_frame: int, end_frame: int, dlc_thresh: float, out_dir_prefix: str = None, export_measurements: bool = False):
    logger.info("Prepare data - Start")
    # We use a redescending cost to stop outliers affecting the optimisation negatively
    redesc_a = 3
    redesc_b = 10
    redesc_c = 20

    t0 = time()

    if out_dir_prefix:
        out_dir = os.path.join(out_dir_prefix, data_dir, "fte_pw")
    else:
        out_dir = os.path.join(root_dir, data_dir, "fte_pw")

    data_dir = os.path.join(root_dir, data_path)
    assert os.path.exists(data_dir)
    dlc_dir = os.path.join(data_dir, "dlc_pw")
    assert os.path.exists(dlc_dir)
    os.makedirs(out_dir, exist_ok=True)

    app.start_logging(os.path.join(out_dir, "fte.log"))

    # load video info
    res, fps, tot_frames, _ = app.get_vid_info(data_dir) # path to original videos
    assert end_frame <= tot_frames, f"end_frame must be less than or equal to {tot_frames}"
    end_frame = tot_frames if end_frame == -1 else end_frame

    start_frame -= 1    # 0 based indexing
    assert start_frame >= 0
    N = end_frame - start_frame
    Ts = 1.0 / fps  # timestep

    logger.info(f"Start frame: {start_frame}, End frame: {end_frame}, Frame rate: {fps}")
    ## ========= POSE FUNCTIONS ========
    pose_to_3d, pos_funcs = data_ops.load_data(os.path.join(root_dir, "pose_3d_functions.pickle"))

    # ========= PROJECTION FUNCTIONS ========
    def pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x*R[0,0] + y*R[0,1] + z*R[0,2] + t.flatten()[0]
        y_2d = x*R[1,0] + y*R[1,1] + z*R[1,2] + t.flatten()[1]
        z_2d = x*R[2,0] + y*R[2,1] + z*R[2,2] + t.flatten()[2]
        #project onto camera plane
        a = x_2d/z_2d
        b = y_2d/z_2d
        #fisheye params
        r = (a**2 + b**2 +1e-12)**0.5
        th = pyo.atan(r)
        #distortion
        th_D = th * (1 + D[0]*th**2 + D[1]*th**4 + D[2]*th**6 + D[3]*th**8)
        x_P = a*th_D/r
        y_P = b*th_D/r
        u = K[0,0]*x_P + K[0,2]
        v = K[1,1]*y_P + K[1,2]
        return u, v

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
        return u

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
        return v

    # ========= IMPORT CAMERA & SCENE PARAMS ========
    K_arr, D_arr, R_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(data_dir)
    D_arr = D_arr.reshape((-1,4))

    # ========= IMPORT DATA ========
    markers = misc.get_markers()

    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        d_idx = {1:"x", 2:"y"}
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val["likelihood"].values[0]

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    # measurement standard deviation
    R = np.array([
        1.2, # nose
        1.24, # l_eye
        1.18, # r_eye
        2.08, # neck_base
        2.04, # spine
        2.52, # tail_base
        2.73, # tail1
        1.83, # tail2
        3.47, # r_shoulder
        2.75, # r_front_knee
        2.69, # r_front_ankle
        # 2.24, # r_front_paw
        3.4, # l_shoulder
        2.91, # l_front_knee
        2.85, # l_front_ankle
        # 2.27, # l_front_paw
        3.26, # r_hip
        2.76, # r_back_knee
        2.33, # r_back_ankle
        # 2.4, # r_back_paw
        3.53, # l_hip
        2.69, # l_back_knee
        2.49, # l_back_ankle
        # 2.34, # l_back_paw
    ], dtype=np.float64)
    # R_pw = np.array([R, [5.13, 3.06, 2.99, 4.07, 5.53, 4.67, 6.05, 5.6, 5.43, 5.39, 6.34, 6.53, 6.14, 6.54, 5.35, 5.33, 6.24, 6.91, 5.8, 6.6],
    # [4.3, 4.72, 4.9, 3.8, 4.4, 5.43, 5.22, 7.29, 5.39, 5.72, 6.01, 6.83, 6.32, 6.27, 5.81, 6.19, 6.22, 7.15, 6.98, 6.5]], dtype=np.float64)
    R_pw = np.array([R, [2.71, 3.06, 2.99, 4.07, 5.53, 4.67, 6.05, 5.6, 5.01, 5.11, 5.24, 5.18, 5.28, 5.5, 4.7, 4.7, 5.21, 5.1, 5.27, 5.75],
    [2.8, 3.24, 3.42, 3.8, 4.4, 5.43, 5.22, 7.29, 8.19, 6.5, 5.9, 8.83, 6.52, 6.22, 6.8, 6.12, 5.37, 7.83, 6.44, 6.1]], dtype=np.float64)
    # R_pw[0, :] = 5
    # R_pw[1, :] = 10
    # R_pw[2, :] = 15
    Q = [ # model parameters variance
        4, 7, 5, # x, y, z
        13, 32, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #  phi_1, ... , phi_14
        9, 18, 43, 53, 90, 118, 247, 186, 194, 164, 295, 243, 334, 149, # theta_1, ... , theta_n
        26, 12, 0, 34, 43, 51, 0, 0, 0, 0, 0, 0, 0, 0, # psi_1, ... , psi_n
    #     ?, ?, ? # lure's x, y, z variance
    ]
    Q = np.array(Q, dtype=np.float64)**2

    #===================================================
    #                   Load in data
    #===================================================
    logger.info("Load H5 2D DLC prediction data")
    df_paths = sorted(glob(os.path.join(dlc_dir, '*.h5')))

    points_2d_df = utils.load_dlc_points_as_df(df_paths, verbose=False)
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df[points_2d_df['likelihood']>dlc_thresh],
        K_arr, D_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )

    # estimate initial points
    logger.info("Estimate the initial trajectory")
    nose_pts = points_3d_df[points_3d_df["marker"]=="nose"][["frame", "x", "y", "z"]].values
    x_slope, x_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,1])
    y_slope, y_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,2])
    z_slope, z_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,3])
    frame_est = np.arange(end_frame)
    x_est = frame_est*x_slope + x_intercept
    y_est = frame_est*y_slope + y_intercept
    z_est = frame_est*z_slope + z_intercept
    psi_est = np.arctan2(y_slope, x_slope)

    logger.info("Prepare data - End")
    #===================================================
    #                   Optimisation
    #===================================================
    logger.info("Setup optimisation - Start")
    m = pyo.ConcreteModel(name = "Cheetah from measurements")
    m.Ts = Ts
    # ===== SETS =====
    P = 3 + 3 * 14 # + 3  # number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n, x_l, y_l, z_l)
    L = len(markers) # number of dlc labels per frame
    C = len(K_arr) # number of cameras
    D2 = 2 # dimensionality of measurements
    D3 = 3 # dimensionality of measurements
    W = 2  # Number of pairwise terms to include + the base measurement.

    m.N = pyo.RangeSet(N)
    m.P = pyo.RangeSet(P)
    m.L = pyo.RangeSet(L)
    m.C = pyo.RangeSet(C)
    m.D2 = pyo.RangeSet(D2)
    m.D3 = pyo.RangeSet(D3)
    m.W = pyo.RangeSet(W)

    # Base measurments. TODO: This is not technically required but it is a lot faster than using pandas for querying data.
    data = {}
    cam_idx = 0
    for path in df_paths:
        dlc_df = pd.read_hdf(path)
        pose_array = dlc_df.droplevel([0], axis=1).to_numpy()
        data[cam_idx] = pose_array
        cam_idx += 1

    # Pairwise correspondence.
    pw_data = {}
    for cam in range(C):
        pw_data[cam] = data_ops.load_data(os.path.join(dlc_dir, f"cam{cam+1}-predictions.pickle"))

    index_dict = {"nose":23, "r_eye":0, "l_eye":1, "neck_base":24, "spine":6, "tail_base":22, "tail1":11,
     "tail2":12, "l_shoulder":13,"l_front_knee":14,"l_front_ankle":15, "l_front_paw": 16, "r_shoulder":2,
      "r_front_knee":3, "r_front_ankle":4, "r_front_paw": 5, "l_hip":17, "l_back_knee":18, "l_back_ankle":19, "l_back_paw": 20,
       "r_hip":7,"r_back_knee":8,"r_back_ankle":9, "r_back_paw": 10}

    # pair_dict = {"r_eye":[23, 24], "l_eye":[23, 24], "nose":[6, 24], "neck_base":[6, 23], "spine":[22, 24], "tail_base":[6, 11], "tail1":[6, 22],
    #  "tail2":[11, 22], "l_shoulder":[6, 24],"l_front_knee":[6, 24],"l_front_ankle":[6, 24],"r_shoulder":[6, 24],
    #   "r_front_knee":[6, 24], "r_front_ankle":[6, 24],"l_hip":[6, 22],"l_back_knee":[6, 22], "l_back_ankle":[6, 22],
    #    "r_hip":[6, 22],"r_back_knee":[6, 22],"r_back_ankle":[6, 22]}
    pair_dict = {"r_eye":[23, 1], "l_eye":[23, 0], "nose":[0, 1], "neck_base":[6, 23], "spine":[22, 24], "tail_base":[6, 11], "tail1":[6, 22],
     "tail2":[11, 22], "l_shoulder":[14, 24],"l_front_knee":[13, 15],"l_front_ankle":[13, 14], "l_front_paw": [14, 15], "r_shoulder":[3, 24],
      "r_front_knee":[2, 4], "r_front_ankle":[2, 3], "r_front_paw": [3, 4], "l_hip": [18, 22], "l_back_knee":[17, 19], "l_back_ankle":[17, 18], "l_back_paw": [18, 19],
       "r_hip":[8, 22], "r_back_knee":[7, 9],"r_back_ankle":[7, 8], "r_back_paw": [8, 9]}

    # ======= WEIGHTS =======
    def init_meas_weights(m, n, c, l, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        marker = markers[l-1]
        if w < 2:
            values = data[c-1][(n-1)+start_frame]
            val = values[2::3]
            base = index_dict[marker]
            likelihood = val[base]
        else:
            base = pair_dict[marker][w-2]
            pw_values = pw_data[c-1][(n-1)+start_frame]
            val = pw_values["pose"][2::3]
            likelihood = val[base]

        # Filter measurements based on DLC threshold. This does ensures that badly predicted points are not considered in the objective function.
        if likelihood > dlc_thresh:
            return 1/R_pw[w-1][l-1]
        else:
            return 0.0
    m.meas_err_weight = pyo.Param(m.N, m.C, m.L, m.W, initialize=init_meas_weights, mutable=True)  # IndexError: index 0 is out of bounds for axis 0 with size 0

    def init_model_weights(m, p):
        if Q[p-1] != 0.0:
            return 1/Q[p-1]
        else:
            return 0.0
    m.model_err_weight = pyo.Param(m.P, initialize=init_model_weights)

    # ===== PARAMETERS =====
    def init_measurements(m, n, c, l, d2, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        values = data[c-1][(n-1)+start_frame]
        val = values[d2-1::3]
        marker = markers[l-1]
        if w < 2:
            base = index_dict[marker]
            return val[base]
        else:
            base = pair_dict[marker][w-2]
            pw_values = pw_data[c-1][(n-1)+start_frame]
            val_pw = pw_values["pws"][:,:,:,d2-1]
            return val[base] + val_pw[0, base, index_dict[marker]]
    m.meas = pyo.Param(m.N, m.C, m.L, m.D2, m.W, initialize=init_measurements)

    if export_measurements:
        # Generate dataframe with the measurements that are used in the optimisation.
        # This allows for inspection of the normal and pairwise predictions used in the FTE.
        measurement_dir = os.path.join(out_dir, "measurements")
        os.makedirs(measurement_dir, exist_ok=True)
        xy_labels = ["x", "y"]
        pd_index = pd.MultiIndex.from_product([markers, xy_labels], names=["bodyparts", "coords"])
        for c in m.C:
            for w in m.W:
                included_measurements = []
                for n in m.N:
                    included_measurements.append([])
                    for l in m.L:
                        if m.meas_err_weight[n, c, l, w] != 0.0:
                            included_measurements[n-1].append([m.meas[n, c, l, 1, w], m.meas[n, c, l, 2, w]])
                        else:
                            included_measurements[n-1].append([float("NaN"), float("NaN")])
                measurements = np.array(included_measurements)
                n_frames = len(measurements)
                df = pd.DataFrame(measurements.reshape((n_frames, -1)), columns=pd_index, index=range(start_frame, start_frame+n_frames))
                # df.to_csv(os.path.join(OUT_DIR, "measurements", f"cam{c}_fte.csv"))
                df.to_hdf(os.path.join(measurement_dir, f"cam{c}_pw_{w}.h5"), "df_with_missing", format="table", mode="w")

    logger.info("Measurement initialisation...Done")
    # ===== VARIABLES =====
    m.x = pyo.Var(m.N, m.P) #position
    m.dx = pyo.Var(m.N, m.P) #velocity
    m.ddx = pyo.Var(m.N, m.P) #acceleration
    m.poses = pyo.Var(m.N, m.L, m.D3)
    m.slack_model = pyo.Var(m.N, m.P)
    m.slack_meas = pyo.Var(m.N, m.C, m.L, m.D2, m.W, initialize=0.0)

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_x[:,0] = x_est[start_frame: start_frame+N] #x # change this to [start_frame: end_frame]?
    init_x[:,1] = y_est[start_frame: start_frame+N] #y
    init_x[:,2] = z_est[start_frame: start_frame+N] #z
    init_x[:,31] = psi_est # yaw = psi
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
    for n in m.N:
        for p in m.P:
            if n<len(init_x): #init using known values
                m.x[n,p].value = init_x[n-1,p-1]
                m.dx[n,p].value = init_dx[n-1,p-1]
                m.ddx[n,p].value = init_ddx[n-1,p-1]
            else: #init using last known value
                m.x[n,p].value = init_x[-1,p-1]
                m.dx[n,p].value = init_dx[-1,p-1]
                m.ddx[n,p].value = init_ddx[-1,p-1]
        #init pose
        var_list = [m.x[n,p].value for p in range(1, P+1)]
        for l in m.L:
            [pos] = pos_funcs[l-1](*var_list)
            # pos = pose_to_3d(*var_list)[l-1]
            for d3 in m.D3:
                m.poses[n,l,d3].value = pos[d3-1]

    logger.info("Variable initialisation...Done")
    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m,n,l,d3):
        #get 3d points
        var_list = [m.x[n,p] for p in range(1, P+1)]
        [pos] = pos_funcs[l-1](*var_list)
        # pos = pose_to_3d(*var_list)[l-1]
        return pos[d3-1] == m.poses[n,l,d3]

    m.pose_constraint = pyo.Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # INTEGRATION
    def backwards_euler_pos(m,n,p): # position
        if n > 1:
    #             return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n-1,p] + m.h**2 * m.ddx[n-1,p]/2
            return m.x[n,p] == m.x[n-1,p] + m.Ts*m.dx[n,p]

        else:
            return pyo.Constraint.Skip
    m.integrate_p = pyo.Constraint(m.N, m.P, rule = backwards_euler_pos)

    def backwards_euler_vel(m,n,p): # velocity
        if n > 1:
            return m.dx[n,p] == m.dx[n-1,p] + m.Ts*m.ddx[n,p]
        else:
            return pyo.Constraint.Skip
    m.integrate_v = pyo.Constraint(m.N, m.P, rule = backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p):
        if n > 1:
            return m.ddx[n,p] == m.ddx[n-1,p] + m.slack_model[n,p]
        else:
            return pyo.Constraint.Skip
    m.constant_acc = pyo.Constraint(m.N, m.P, rule = constant_acc)

    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2, w):
        #project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        x, y, z = m.poses[n,l,1], m.poses[n,l,2], m.poses[n,l,3]
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2, w] - m.slack_meas[n, c, l, d2, w] == 0.0
    m.measurement = pyo.Constraint(m.N, m.C, m.L, m.D2, m.W, rule = measurement_constraints)

    #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
    #Head
    def head_psi_0(m,n):
        return abs(m.x[n,4]) <= np.pi/6
    m.head_psi_0 = pyo.Constraint(m.N, rule=head_psi_0)
    def head_theta_0(m,n):
        return abs(m.x[n,18]) <= np.pi/6
    m.head_theta_0 = pyo.Constraint(m.N, rule=head_theta_0)

    #Neck
    def neck_phi_1(m,n):
        return abs(m.x[n,5]) <= np.pi/6
    m.neck_phi_1 = pyo.Constraint(m.N, rule=neck_phi_1)
    def neck_theta_1(m,n):
        return abs(m.x[n,19]) <= np.pi/6
    m.neck_theta_1 = pyo.Constraint(m.N, rule=neck_theta_1)
    def neck_psi_1(m,n):
        return abs(m.x[n,33]) <= np.pi/6
    m.neck_psi_1 = pyo.Constraint(m.N, rule=neck_psi_1)

    #Front torso
    def front_torso_theta_2(m,n):
        return abs(m.x[n,20]) <= np.pi/6
    m.front_torso_theta_2 = pyo.Constraint(m.N, rule=front_torso_theta_2)

    #Back torso
    def back_torso_theta_3(m,n):
        return abs(m.x[n,21]) <= np.pi/6
    m.back_torso_theta_3 = pyo.Constraint(m.N, rule=back_torso_theta_3)
    def back_torso_phi_3(m,n):
        return abs(m.x[n,7]) <= np.pi/6
    m.back_torso_phi_3 = pyo.Constraint(m.N, rule=back_torso_phi_3)
    def back_torso_psi_3(m,n):
        return abs(m.x[n,35]) <= np.pi/6
    m.back_torso_psi_3 = pyo.Constraint(m.N, rule=back_torso_psi_3)

    #Tail base
    def tail_base_theta_4(m,n):
        return abs(m.x[n,22]) <= np.pi/1.5
    m.tail_base_theta_4 = pyo.Constraint(m.N, rule=tail_base_theta_4)
    def tail_base_psi_4(m,n):
        return abs(m.x[n,36]) <= np.pi/1.5
    m.tail_base_psi_4 = pyo.Constraint(m.N, rule=tail_base_psi_4)

    #Tail mid
    def tail_mid_theta_5(m,n):
        return abs(m.x[n,23]) <= np.pi/1.5
    m.tail_mid_theta_5 = pyo.Constraint(m.N, rule=tail_mid_theta_5)
    def tail_mid_psi_5(m,n):
        return abs(m.x[n,37]) <= np.pi/1.5
    m.tail_mid_psi_5 = pyo.Constraint(m.N, rule=tail_mid_psi_5)

    #Front left leg
    def l_shoulder_theta_6(m,n):
        return abs(m.x[n,24]) <= np.pi/2
    m.l_shoulder_theta_6 = pyo.Constraint(m.N, rule=l_shoulder_theta_6)
    def l_front_knee_theta_7(m,n):
        return abs(m.x[n,25] + np.pi/2) <= np.pi/2
    m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=l_front_knee_theta_7)

    #Front right leg
    def r_shoulder_theta_8(m,n):
        return abs(m.x[n,26]) <= np.pi/2
    m.r_shoulder_theta_8 = pyo.Constraint(m.N, rule=r_shoulder_theta_8)
    def r_front_knee_theta_9(m,n):
        return abs(m.x[n,27] + np.pi/2) <= np.pi/2
    m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=r_front_knee_theta_9)

    #Back left leg
    def l_hip_theta_10(m,n):
        return abs(m.x[n,28]) <= np.pi/2
    m.l_hip_theta_10 = pyo.Constraint(m.N, rule=l_hip_theta_10)
    def l_back_knee_theta_11(m,n):
        return abs(m.x[n,29] - np.pi/2) <= np.pi/2
    m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=l_back_knee_theta_11)

    #Back right leg
    def r_hip_theta_12(m,n):
        return abs(m.x[n,30]) <= np.pi/2
    m.r_hip_theta_12 = pyo.Constraint(m.N, rule=r_hip_theta_12)
    def r_back_knee_theta_13(m,n):
        return abs(m.x[n,31] - np.pi/2) <= np.pi/2
    m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=r_back_knee_theta_13)

    logger.info("Constaint initialisation...Done")
    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0
        for n in m.N:
            #Model Error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p] ** 2
            #Measurement Error
            for l in m.L:
                for c in m.C:
                    for d2 in m.D2:
                        # slack_meas_err += misc.redescending_loss(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2], redesc_a, redesc_b, redesc_c)
                        for w in m.W:
                            slack_meas_err += misc.redescending_loss(m.meas_err_weight[n, c, l, w] * m.slack_meas[n, c, l, d2, w], redesc_a, redesc_b, redesc_c)
        return slack_meas_err + slack_model_err

    m.obj = pyo.Objective(rule = obj)

    logger.info("Objective initialisation...Done")
    # RUN THE SOLVER
    opt = SolverFactory(
        'ipopt',
        executable='/home/zico/lib/ipopt/build/bin/ipopt'
    )

    # solver options
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 10000
    opt.options["max_cpu_time"] = 10000
    opt.options["tol"] = 1e-1
    opt.options["OF_print_timing_statistics"] = "yes"
    opt.options["OF_print_frequency_time"] = 10
    opt.options["OF_hessian_approximation"] = "limited-memory"
    opt.options["linear_solver"] = "ma86"

    logger.info("Setup optimisation - End")
    t1 = time()
    logger.info(f"Initialisation took {t1 - t0:.2f}s")

    t0 = time()
    results = opt.solve(m, tee=True)
    t1 = time()
    logger.info(f"Optimisation solver took {t1 - t0:.2f}s")

    app.stop_logging()

    logger.info("Generate outputs...")
    # ===== SAVE FTE RESULTS =====
    def convert_m(m, pose_indices):
        x_optimised, dx_optimised, ddx_optimised = [], [], []
        for n in m.N:
            x_optimised.append([pyo.value(m.x[n, p]) for p in m.P])
            dx_optimised.append([pyo.value(m.dx[n, p]) for p in m.P])
            ddx_optimised.append([pyo.value(m.ddx[n, p]) for p in m.P])

        positions = [pose_to_3d(*states) for states in x_optimised]

        # remove zero-valued vars
        for n in m.N:
            n -= 1 # remember pyomo's 1-based indexing
            for p in pose_indices[::-1]:
                    assert x_optimised[n][p] == 0
                    del x_optimised[n][p]
                    del dx_optimised[n][p]
                    del ddx_optimised[n][p]

        states = dict(
            x=x_optimised,
            dx=dx_optimised,
            ddx=ddx_optimised,
        )
        return positions, states

    [unused_pose_indices] = np.where(Q == 0)
    positions, states = convert_m(m, unused_pose_indices)

    out_fpath = os.path.join(out_dir, f"fte.pickle")
    app.save_optimised_cheetah(positions, out_fpath, extra_data=dict(**states, start_frame=start_frame))
    app.save_3d_cheetah_as_2d(positions, out_dir, scene_fpath, markers, project_points_fisheye, start_frame)

    logger.info("Done")
