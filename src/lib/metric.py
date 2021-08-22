import numpy as np
import pandas as pd
from typing import List, Dict, Union

from . import calib


def residual_error(points_2d_df, points_3d_dfs, markers, camera_params) -> Dict:
    k_arr, d_arr, r_arr, t_arr, _, _ = camera_params
    n_cam = len(k_arr)
    if not isinstance(points_3d_dfs, list):
        points_3d_dfs = [points_3d_dfs] * n_cam
    error = {str(i): None for i in range(n_cam)}
    for i in range(n_cam):
        error_dfs = []
        for m in markers:
            # extract frames
            q = f'marker == "{m}"'
            pts_2d_df = points_2d_df.query(q + f'and camera == {i}')
            pts_3d_df = points_3d_dfs[i].query(q)
            pts_2d_df = pts_2d_df[pts_2d_df[['x', 'y']].notnull().all(axis=1)]
            pts_3d_df = pts_3d_df[pts_3d_df[['x', 'y', 'z']].notnull().all(axis=1)]
            valid_frames = np.intersect1d(pts_2d_df['frame'].to_numpy(), pts_3d_df['frame'].to_numpy())
            pts_2d_df = pts_2d_df[pts_2d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            pts_3d_df = pts_3d_df[pts_3d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])

            # get 2d and reprojected points
            frames = pts_2d_df['frame'].to_numpy()
            pts_2d = pts_2d_df[['x', 'y']].to_numpy()
            pts_3d = pts_3d_df[['x', 'y', 'z']].to_numpy()
            if len(pts_2d) == 0 or len(pts_3d) == 0:
                continue
            prj_2d = calib.project_points_fisheye(pts_3d, k_arr[i], d_arr[i], r_arr[i], t_arr[i])

            # camera distance
            cam_pos = np.squeeze(t_arr[i, :, :])
            cam_dist = np.sqrt(np.sum((pts_3d - cam_pos)**2, axis=1))

            # compare both types of points
            residual = np.sqrt(np.sum((pts_2d - prj_2d)**2, axis=1))
            error_uv = pts_2d - prj_2d

            # make the result dataframe
            marker_arr = np.array([m] * len(frames))
            error_dfs.append(
                pd.DataFrame(np.vstack((frames, marker_arr, cam_dist, residual, error_uv.T)).T,
                             columns=['frame', 'marker', 'camera_distance', 'pixel_residual', 'error_u', 'error_v']))

        error[str(i)] = pd.concat(error_dfs, ignore_index=True) if len(error_dfs) > 0 else pd.DataFrame(
            columns=['frame', 'marker', 'camera_distance', 'pixel_residual', 'error_u', 'error_v'])

    return error
