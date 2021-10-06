import sys
import numpy as np
import pandas as pd
import sympy as sp

from py_utils import log

# Create a module logger with the name of this file.
logger = log.logger(__name__)


def get_markers():
    return [
        'nose',
        'r_eye',
        'l_eye',
        'neck_base',
        'spine',
        'tail_base',
        'tail1',
        'tail2',
        'r_shoulder',
        'r_front_knee',
        'r_front_ankle',
        'r_front_paw',
        'l_shoulder',
        'l_front_knee',
        'l_front_ankle',
        'l_front_paw',
        'r_hip',
        'r_back_knee',
        'r_back_ankle',
        'r_back_paw',
        'l_hip',
        'l_back_knee',
        'l_back_ankle',
        'l_back_paw',
    ]


def get_dlc_marker_indices():
    return {
        "nose": 23,
        "r_eye": 0,
        "l_eye": 1,
        "neck_base": 24,
        "spine": 6,
        "tail_base": 22,
        "tail1": 11,
        "tail2": 12,
        "l_shoulder": 13,
        "l_front_knee": 14,
        "l_front_ankle": 15,
        "l_front_paw": 16,
        "r_shoulder": 2,
        "r_front_knee": 3,
        "r_front_ankle": 4,
        "r_front_paw": 5,
        "l_hip": 17,
        "l_back_knee": 18,
        "l_back_ankle": 19,
        "l_back_paw": 20,
        "r_hip": 7,
        "r_back_knee": 8,
        "r_back_ankle": 9,
        "r_back_paw": 10
    }


def get_skeleton():
    return [['nose', 'l_eye'], ['nose', 'r_eye'], ['nose', 'neck_base'], ['l_eye', 'neck_base'], ['r_eye', 'neck_base'],
            ['neck_base', 'spine'], ['spine', 'tail_base'], ['tail_base', 'tail1'], ['tail1', 'tail2'],
            ['neck_base', 'r_shoulder'], ['r_shoulder', 'r_front_knee'], ['r_front_knee', 'r_front_ankle'],
            ['r_front_ankle', 'r_front_paw'], ['neck_base', 'l_shoulder'], ['l_shoulder', 'l_front_knee'],
            ['l_front_knee', 'l_front_ankle'], ['l_front_ankle', 'l_front_paw'], ['tail_base', 'r_hip'],
            ['r_hip', 'r_back_knee'], ['r_back_knee', 'r_back_ankle'], ['r_back_ankle', 'r_back_paw'],
            ['tail_base', 'l_hip'], ['l_hip', 'l_back_knee'], ['l_back_knee', 'l_back_ankle'],
            ['l_back_ankle', 'l_back_paw']]


def get_pose_params():
    states = [
        'x_0',
        'y_0',
        'z_0',  # head position in inertial
        'phi_0',
        'theta_0',
        'psi_0',  # head rotation in inertial
        'phi_1',
        'theta_1',
        'psi_1',  # neck
        'theta_2',  # front torso
        'phi_3',
        'theta_3',
        'psi_3',  # back torso
        'theta_4',
        'psi_4',  # tail_base
        'theta_5',
        'psi_5',  # tail_mid
        'theta_6',
        'theta_7',  # l_shoulder, l_front_knee
        'theta_8',
        'theta_9',  # r_shoulder, r_front_knee
        'theta_10',
        'theta_11',  # l_hip, l_back_knee
        'theta_12',
        'theta_13',  # r_hip, r_back_knee
        'theta_14',  # l_front_ankle
        'theta_15',  # r_front_ankle
        'theta_16',  # l_back_ankle
        'theta_17',  # r_back_ankle
    ]
    return dict(zip(states, range(len(states))))


def get_pairwise_graph():
    return {
        "r_eye": [23, 1],
        "l_eye": [23, 0],
        "nose": [0, 1],
        "neck_base": [6, 23],
        "spine": [22, 24],
        "tail_base": [6, 11],
        "tail1": [6, 22],
        "tail2": [11, 22],
        "l_shoulder": [14, 24],
        "l_front_knee": [13, 15],
        "l_front_ankle": [13, 14],
        "l_front_paw": [14, 15],
        "r_shoulder": [3, 24],
        "r_front_knee": [2, 4],
        "r_front_ankle": [2, 3],
        "r_front_paw": [3, 4],
        "l_hip": [18, 22],
        "l_back_knee": [17, 19],
        "l_back_ankle": [17, 18],
        "l_back_paw": [18, 19],
        "r_hip": [8, 22],
        "r_back_knee": [7, 9],
        "r_back_ankle": [7, 8],
        "r_back_paw": [8, 9]
    }
    # return {
    #     "r_eye": [23, 24],
    #     "l_eye": [23, 24],
    #     "nose": [6, 24],
    #     "neck_base": [6, 23],
    #     "spine": [22, 24],
    #     "tail_base": [6, 11],
    #     "tail1": [6, 22],
    #     "tail2": [11, 22],
    #     "l_shoulder": [6, 24],
    #     "l_front_knee": [6, 24],
    #     "l_front_ankle": [6, 24],
    #     "r_shoulder": [6, 24],
    #     "r_front_knee": [6, 24],
    #     "r_front_ankle": [6, 24],
    #     "l_hip": [6, 22],
    #     "l_back_knee": [6, 22],
    #     "l_back_ankle": [6, 22],
    #     "r_hip": [6, 22],
    #     "r_back_knee": [6, 22],
    #     "r_back_ankle": [6, 22]
    # }


def get_3d_marker_coords(x):
    """Returns either a numpy array or a sympy Matrix of the 3D marker coordinates (shape Nx3) for a given state vector x.
    """
    idx = get_pose_params()

    func = sp.Matrix if isinstance(x[0], sp.Expr) else np.array

    # rotations
    RI_0 = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])  # head
    R0_I = RI_0.T
    RI_1 = rot_z(x[idx['psi_1']]) @ rot_x(x[idx['phi_1']]) @ rot_y(x[idx['theta_1']]) @ RI_0  # neck
    R1_I = RI_1.T
    RI_2 = rot_y(x[idx['theta_2']]) @ RI_1  # front torso
    R2_I = RI_2.T
    RI_3 = rot_z(x[idx['psi_3']]) @ rot_x(x[idx['phi_3']]) @ rot_y(x[idx['theta_3']]) @ RI_2  # back torso
    R3_I = RI_3.T
    RI_4 = rot_z(x[idx['psi_4']]) @ rot_y(x[idx['theta_4']]) @ RI_3  # tail base
    R4_I = RI_4.T
    RI_5 = rot_z(x[idx['psi_5']]) @ rot_y(x[idx['theta_5']]) @ RI_4  # tail mid
    R5_I = RI_5.T
    RI_6 = rot_y(x[idx['theta_6']]) @ RI_2  # l_shoulder
    R6_I = RI_6.T
    RI_7 = rot_y(x[idx['theta_7']]) @ RI_6  # l_front_knee
    R7_I = RI_7.T
    RI_8 = rot_y(x[idx['theta_8']]) @ RI_2  # r_shoulder
    R8_I = RI_8.T
    RI_9 = rot_y(x[idx['theta_9']]) @ RI_8  # r_front_knee
    R9_I = RI_9.T
    RI_10 = rot_y(x[idx['theta_10']]) @ RI_3  # l_hip
    R10_I = RI_10.T
    RI_11 = rot_y(x[idx['theta_11']]) @ RI_10  # l_back_knee
    R11_I = RI_11.T
    RI_12 = rot_y(x[idx['theta_12']]) @ RI_3  # r_hip
    R12_I = RI_12.T
    RI_13 = rot_y(x[idx['theta_13']]) @ RI_12  # r_back_knee
    R13_I = RI_13.T
    RI_14 = rot_y(x[idx["theta_14"]]) @ RI_7  # l_front_ankle
    R14_I = RI_14.T
    RI_15 = rot_y(x[idx["theta_15"]]) @ RI_9  # r_front_ankle
    R15_I = RI_15.T
    RI_16 = rot_y(x[idx["theta_16"]]) @ RI_11  # l_back_ankle
    R16_I = RI_16.T
    RI_17 = rot_y(x[idx["theta_17"]]) @ RI_13  # r_back_ankle
    R17_I = RI_17.T

    # positions
    p_head = func([x[idx['x_0']], x[idx['y_0']], x[idx['z_0']]])

    p_l_eye = p_head + R0_I @ func([0, 0.03, 0])
    p_r_eye = p_head + R0_I @ func([0, -0.03, 0])
    p_nose = p_head + R0_I @ func([0.055, 0, -0.055])

    p_neck_base = p_head + R1_I @ func([-0.28, 0, 0])
    p_spine = p_neck_base + R2_I @ func([-0.37, 0, 0])

    p_tail_base = p_spine + R3_I @ func([-0.37, 0, 0])
    p_tail_mid = p_tail_base + R4_I @ func([-0.28, 0, 0])
    p_tail_tip = p_tail_mid + R5_I @ func([-0.36, 0, 0])

    p_l_shoulder = p_neck_base + R2_I @ func([-0.04, 0.08, -0.10])
    p_l_front_knee = p_l_shoulder + R6_I @ func([0, 0, -0.24])
    p_l_front_ankle = p_l_front_knee + R7_I @ func([0, 0, -0.28])
    p_l_front_paw = p_l_front_ankle + R14_I @ func([0, 0, -0.14])

    p_r_shoulder = p_neck_base + R2_I @ func([-0.04, -0.08, -0.10])
    p_r_front_knee = p_r_shoulder + R8_I @ func([0, 0, -0.24])
    p_r_front_ankle = p_r_front_knee + R9_I @ func([0, 0, -0.28])
    p_r_front_paw = p_r_front_ankle + R15_I @ func([0, 0, -0.14])

    p_l_hip = p_tail_base + R3_I @ func([0.12, 0.08, -0.06])
    p_l_back_knee = p_l_hip + R10_I @ func([0, 0, -0.32])
    p_l_back_ankle = p_l_back_knee + R11_I @ func([0, 0, -0.25])
    p_l_back_paw = p_l_back_ankle + R16_I @ func([0, 0, -0.22])

    p_r_hip = p_tail_base + R3_I @ func([0.12, -0.08, -0.06])
    p_r_back_knee = p_r_hip + R12_I @ func([0, 0, -0.32])
    p_r_back_ankle = p_r_back_knee + R13_I @ func([0, 0, -0.25])
    p_r_back_paw = p_r_back_ankle + R17_I @ func([0, 0, -0.22])

    return func([
        p_nose.T, p_r_eye.T, p_l_eye.T, p_neck_base.T, p_spine.T, p_tail_base.T, p_tail_mid.T, p_tail_tip.T,
        p_r_shoulder.T, p_r_front_knee.T, p_r_front_ankle.T, p_r_front_paw.T, p_l_shoulder.T, p_l_front_knee.T,
        p_l_front_ankle.T, p_l_front_paw.T, p_r_hip.T, p_r_back_knee.T, p_r_back_ankle.T, p_r_back_paw.T, p_l_hip.T,
        p_l_back_knee.T, p_l_back_ankle.T, p_l_back_paw.T
    ])


def redescending_loss(err, a, b, c) -> float:
    # outlier rejecting cost function
    def func_step(start, x):
        return 1 / (1 + np.e**(-1 * (x - start)))

    def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)

    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e)) / 2 * e**2
    cost += func_piece(a, b, e) * (a * e - (a**2) / 2)
    cost += func_piece(b, c, e) * (a * b - (a**2) / 2 + (a * (c - b) / 2) * (1 - ((c - e) / (c - b))**2))
    cost += func_step(c, e) * (a * b - (a**2) / 2 + (a * (c - b) / 2))
    return cost


def redescending_smooth_loss(r, c, arctan_func) -> float:
    cost = (0.25 * c**2 * (arctan_func(r / c)**2 + ((c * r)**2) / (c**4 + r**4)))
    return cost


def cauchy_loss(r, c, log_func) -> float:
    cost = c**2 * (log_func(1 + (r / c)**2))
    return cost


def fair_loss(r, c, log_func) -> float:
    cost = (c**2 * ((abs(r) / c) - log_func(1 + (abs(r) / c))))
    return cost


def global_positions(R_arr, t_arr):
    "Returns a vector of camera position vectors in the world frame"
    R_arr = np.array(R_arr).reshape((-1, 3, 3))
    t_arr = np.array(t_arr).reshape((-1, 3, 1))

    positions = []
    assert R_arr.shape[0] == t_arr.shape[0], 'Number of cams in R_arr do not match t_arr'
    for r, t in zip(R_arr, t_arr):
        pos = -r.T @ t
        positions.append(pos)

    return np.array(positions, dtype=np.float32)


def rotation_matrix_from_vectors(u, v):
    """ Find the rotation matrix that aligns u to v
    :param u: A 3D "source" vector
    :param v: A 3D "destination" vector
    :return mat: A transform matrix (3x3) which when applied to u, aligns it with v.
    """
    # https://stackoverflow.com/questions/36409140/create-a-rotation-matrix-from-2-normals
    # Suppose you want to write the rotation that maps a vector u to a vector v.
    # if U and V are their unit vectors then W = U^V (cross product) is the axis of rotation and is an invariant
    # Let M be the associated matrix.
    # We have finally: (V,W,V^W) = M.(U,W,U^W)

    U = (u / np.linalg.norm(u)).reshape(3)
    V = (v / np.linalg.norm(v)).reshape(3)

    W = np.cross(U, V)
    A = np.array([U, W, np.cross(U, W)]).T
    B = np.array([V, W, np.cross(V, W)]).T
    return np.dot(B, np.linalg.inv(A))


def rot_x(x):
    if isinstance(x, sp.Expr):
        c = sp.cos(x)
        s = sp.sin(x)
        func = sp.Matrix
    else:
        c = np.cos(x)
        s = np.sin(x)
        func = np.array
    return func([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_y(y):
    if isinstance(y, sp.Expr):
        c = sp.cos(y)
        s = sp.sin(y)
        func = sp.Matrix
    else:
        c = np.cos(y)
        s = np.sin(y)
        func = np.array
    return func([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rot_z(z):
    if isinstance(z, sp.Expr):
        c = sp.cos(z)
        s = sp.sin(z)
        func = sp.Matrix
    else:
        c = np.cos(z)
        s = np.sin(z)
        func = np.array
    return func([[c, s, 0], [-s, c, 0], [0, 0, 1]])


class PoseReduction:
    def __init__(self, dataset_fname: str, pose_params: dict, ext_dim: int, n_comps: int, standardise: bool = False):
        self.p_idx = pose_params
        self.num_vars = len(pose_params.keys())
        self.ext_dim = ext_dim
        self.standardise = standardise
        self.included_vars = np.arange(ext_dim, self.num_vars, dtype=int)
        self.excluded_vars = np.arange(0, ext_dim, dtype=int)

        df = pd.read_hdf(dataset_fname)
        # Swap columns (only used if 4 parameters from ext_dim)
        if self.ext_dim == 4:
            col_list = df.columns.tolist()
            df = df[self._swap_yaw_roll_angles(np.asarray(col_list))]
            self.excluded_vars = np.array(
                [self.p_idx['x_0'], self.p_idx['y_0'], self.p_idx['z_0'], self.p_idx['psi_0']], dtype=int)
            self.included_vars = np.arange(0, self.num_vars, dtype=int)
            self.included_vars = np.delete(self.included_vars, self.excluded_vars)

        X = df.iloc[:, self.ext_dim:self.num_vars].to_numpy()
        self.std = X.std(axis=0)
        self.mean = X.mean(axis=0)
        if standardise:
            X0 = (X - self.mean) / self.std
        else:
            X0 = X - self.mean

        U, s, VT = np.linalg.svd(X0, full_matrices=False)
        # Sign correction to ensure deterministic output from SVD. Code taken from sklearn.
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        VT *= signs[:, np.newaxis]

        # Calcuate the explained variance and determine the covariance matrix from singular values.
        eig_values = s**2
        variance_explained = np.cumsum(eig_values) / np.sum(eig_values)
        self.S = np.diag(s)
        # COV = S @ S

        # Obtain the principal axes (i.e. new basis vectors) and place in a projection matrix.
        self.P = VT[:n_comps, :]
        # Get prinical components of dataset.
        self.PC = U[:, :n_comps] * s[:n_comps]
        if standardise:
            X1 = self.PC.dot(self.P) * self.std + self.mean
        else:
            X1 = self.PC.dot(self.P) + self.mean

        # Calculate reconstruction error and error variance.
        X_orig = df.iloc[:, :self.num_vars].to_numpy()
        self.rmse = np.sqrt(np.mean((X_orig[:, self.ext_dim:] - X1)**2, axis=0))
        self.error_variance = np.zeros(self.num_vars)
        var = np.var(X_orig[:, self.ext_dim:] - X1, axis=0)
        self.error_variance[self.included_vars] = var

        reconstructed_states = np.concatenate((X_orig[:, :self.ext_dim], X1), axis=1)
        positions_orig = np.array([get_3d_marker_coords(pose) for pose in X_orig])
        positions_pca = np.array([get_3d_marker_coords(pose) for pose in reconstructed_states])
        position_diff = (positions_orig - positions_pca)
        mpjpe_mm = np.mean(np.sqrt(np.sum(position_diff**2, axis=2)), axis=0) * 1000.0
        logger.info(
            f'PCA trained for {n_comps} components ({np.round(100.0 * variance_explained[n_comps - 1], 2)}%) with reconstruction error [mm]: {mpjpe_mm.mean(axis=0).round(4)}'
        )

    def pc_std(self):
        return np.std(self.PC, axis=0)

    def project(self, X: np.ndarray, full_state: bool = True, inverse: bool = False) -> np.ndarray:
        if full_state:
            X_full = X.copy()
            if len(X.shape) > 1:
                mask = np.ones(X.shape[1], dtype=bool)
                mask[self.excluded_vars] = False
                X = X[:, mask]
            else:
                mask = np.ones(X.size, dtype=bool)
                mask[self.excluded_vars] = False
                X = X[mask]
            if self.standardise:
                if inverse: return self._get_full_pose(X_full, np.dot(X, self.P) * self.std + self.mean)
                return self._get_full_pose(X_full, np.dot((X - self.mean) / self.std, self.P.T))
            else:
                if inverse: return self._get_full_pose(X_full, np.dot(X, self.P) + self.mean)
                return self._get_full_pose(X_full, np.dot(X - self.mean, self.P.T))

        if self.standardise:
            if inverse: return np.dot(X, self.P) * self.std + self.mean
            return np.dot((X - self.mean) / self.std, self.P.T)
        else:
            if inverse: return np.dot(X, self.P) + self.mean
            return np.dot(X - self.mean, self.P.T)

    def _get_full_pose(self, X: np.ndarray, X_reduced: np.ndarray) -> np.ndarray:
        if len(X.shape) > 1:
            ret = np.concatenate((X[:, self.excluded_vars], X_reduced), axis=1)
        else:
            ret = np.concatenate((X[self.excluded_vars], X_reduced), axis=0)

        return self._swap_yaw_roll_angles(ret) if self.ext_dim == 4 else ret

    def _swap_yaw_roll_angles(self, X: np.ndarray):
        if len(X.shape) > 1:
            X[:, self.p_idx['phi_0']], X[:, self.p_idx['psi_0']] = X[:, self.p_idx['psi_0']], X[:, self.p_idx['phi_0']]
        else:
            X[self.p_idx['phi_0']], X[self.p_idx['psi_0']] = X[self.p_idx['psi_0']], X[self.p_idx['phi_0']]

        return X


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger:
    def __init__(self, out_fpath):
        self.terminal = sys.stdout
        self.logfile = open(out_fpath, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
