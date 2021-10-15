import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics

from lib import misc

from py_utils import log, data_ops

# Create a module logger with the name of this file.
logger = log.logger(__name__)


class PoseModel:
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
        self.covar = np.dot(VT.T, self.S)

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
        positions_orig = np.array([misc.get_3d_marker_coords(pose) for pose in X_orig])
        positions_pca = np.array([misc.get_3d_marker_coords(pose) for pose in reconstructed_states])
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


class MotionModel:
    def __init__(self,
                 dataset_fname: str,
                 num_params: int,
                 start_idx: int = 0,
                 window_size: int = 10,
                 window_time: int = 1):
        # Preprocessing step to prepare the dataset for Linear Regression fit.
        self.window_size = window_size
        self.window_time = window_time
        df = pd.read_hdf(dataset_fname)
        idx = np.where(df.index.values == 0)[0]
        df_in = df.iloc[:, start_idx:start_idx + num_params]
        df_list = []
        end_segment = 0
        for begin_segment, end_segment in zip(idx, idx[1:]):
            df_list.append(
                data_ops.series_to_supervised(df_in.iloc[begin_segment:end_segment],
                                              n_in=window_size,
                                              n_step=window_time))
        df_list.append(data_ops.series_to_supervised(df_in.iloc[end_segment:], n_in=window_size, n_step=window_time))
        df_input = pd.concat(df_list)

        # Instantiate the LR model and split the dataset into train and test sets.
        self.lr_model = LinearRegression()
        # self.lr_model = Lasso(alpha=0.01, random_state=42, max_iter=5000)
        xy_set = df_input.to_numpy()
        X = xy_set[:, 0:(num_params * window_size)]
        y = xy_set[:, (num_params * window_size):]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        logger.info(f"Motion model X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Motion model X_test: {X_test.shape}, y_test: {y_test.shape}")
        self.lr_model.fit(X_train, y_train)
        y_pred = self.lr_model.predict(X_test)

        # Determine the error for the test set.
        residuals = y_test - y_pred
        self.error_variance = np.var(residuals, axis=0)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        explained_variance = metrics.explained_variance_score(y_test, y_pred)
        max_error = metrics.max_error(y_test.flatten(), y_pred.flatten())

        logger.info(
            f"Model RMSE: {rmse:.6f}, Max Error: {max_error:.6f}, Explained variance: {100*explained_variance:.2f}%")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(self.lr_model.coef_, X.flatten()) + self.lr_model.intercept_
