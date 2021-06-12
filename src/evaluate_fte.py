import os
import glob
from typing import Dict
from argparse import ArgumentParser

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cycler

from py_utils import data_ops
from lib import misc

plt.rcParams['axes.grid'] = True


def _calculate_delta_acc(acc: np.ndarray) -> np.ndarray:
    return np.array([(acc[n, :] - acc[n - 1, :]) for n in range(1, len(acc))])


def eval_delta_acc(data: Dict, results_dir: str, show_plot=False) -> np.ndarray:
    start_frame = data["start_frame"]
    acc = np.array(data["ddx"])
    x_axis_range = range(start_frame, start_frame + len(acc))
    delta_acc = _calculate_delta_acc(acc)

    fig = plt.figure(figsize=(16, 12), dpi=120)
    fig.suptitle("Acceleration and Delta Acceleration (-) States", fontsize=14)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.subplot(321)
    plt.plot(x_axis_range, acc[:, 17], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 17], "--b")
    plt.title("Left Shoulder")

    plt.subplot(322)
    plt.plot(x_axis_range, acc[:, 0], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 0], "--b")
    plt.title("X")

    plt.subplot(323)
    plt.plot(x_axis_range, acc[:, 23], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 1], "--b")
    plt.title("Right Hip")

    plt.subplot(324)
    plt.plot(x_axis_range, acc[:, 1], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 1], "--b")
    plt.title("Y")

    plt.subplot(325)
    plt.plot(x_axis_range, acc[:, 21], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 21], "--b")
    plt.title("Left Hip")

    plt.subplot(326)
    plt.plot(x_axis_range, acc[:, 2], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 2], "--b")
    plt.title("Z")

    # Set common labels
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.04, 0.5, "Acceleration [m/s^2]", ha='center', va='center', rotation='vertical')

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(results_dir, "fte_acceleration.png"))
        plt.close()

    return np.max(np.abs(delta_acc), axis=0)


def eval_meas_error(data: Dict, results_dir: str, show_plot=False) -> None:
    start_frame = data["start_frame"]
    meas_err = data["meas_err"]
    meas_weight = data["meas_weight"]

    x_axis_range = range(start_frame, start_frame + len(meas_weight))

    num_cams = meas_err.shape[1]
    single_view = False
    if len(meas_weight.shape) < 3:
        single_view = True
    meas_weight = np.expand_dims(meas_weight, 3) if len(meas_weight.shape) > 2 else np.expand_dims(meas_weight, 2)
    weighted_meas_err = meas_weight * meas_err

    if single_view:
        num_cams = 1
        xy_meas_err = np.mean(meas_err, axis=2)
        xy_filtered_meas_err = np.mean(weighted_meas_err, axis=2)
        xy_meas_err = np.expand_dims(xy_meas_err, axis=0)
        xy_filtered_meas_err = np.expand_dims(xy_filtered_meas_err, axis=0)
    else:
        xy_meas_err = np.array([np.mean(meas_err[:, cam_idx], axis=2) for cam_idx in range(num_cams)])
        xy_filtered_meas_err = np.array([np.mean(weighted_meas_err[:, cam_idx], axis=2) for cam_idx in range(num_cams)])

    markers = misc.get_markers()
    marker_colors = cm.jet(np.linspace(0, 1, len(markers)))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', marker_colors)
    fig = plt.figure(figsize=(16, 12), dpi=120)
    fig.suptitle("Reprojection Error (Before Filtering and Scaling)", fontsize=14)
    base_subplot_value = 320
    plotted_values = None
    for idx in range(num_cams):
        base_subplot_value += 1
        plt.subplot(base_subplot_value)
        plt.title(f"CAM {idx+1}")
        plotted_values = plt.plot(x_axis_range, xy_meas_err[idx, :, :], marker="o", markersize=2)

    # Set common labels
    fig.legend(plotted_values, markers, loc=(0.91, 0.4))
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.06, 0.5, "Error [pixels]", ha='center', va='center', rotation='vertical')

    if not show_plot:
        plt.savefig(os.path.join(results_dir, "fte_meas_error.png"))
        plt.close()

    fig = plt.figure(figsize=(16, 12), dpi=120)
    fig.suptitle("Reprojection Error (After Filtering and Scaling)", fontsize=14)
    base_subplot_value = 320
    for idx in range(num_cams):
        base_subplot_value += 1
        plt.subplot(base_subplot_value)
        plt.title(
            f"CAM {idx+1} (\u03BC: {np.mean(xy_filtered_meas_err[idx, :, :]):.2f}, \u03C3: {np.std(xy_filtered_meas_err[idx, :, :]):.2f})"
        )
        plotted_values = plt.plot(x_axis_range, xy_filtered_meas_err[idx, :, :], marker="o", markersize=2)

    # Set common labels
    fig.legend(plotted_values, markers, loc=(0.91, 0.4))
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.06, 0.5, "Error [pixels]", ha='center', va='center', rotation='vertical')

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(results_dir, "fte_meas_error_filtered.png"))
        plt.close()


def eval_model_error(data: Dict, results_dir: str, show_plot=False) -> None:
    start_frame = data["start_frame"]
    model_err = data["model_err"]
    model_weight = data["model_weight"]

    # Plot the x, y, z states together with the l_shoulder and r_hip.
    state_indices = {0: "x", 1: "y", 2: "z", 22: "l_shoulder", 28: "r_hip"}
    x_axis_range = range(start_frame, start_frame + len(model_err))
    avg_model_err = np.mean(model_err[:, model_weight != 0], axis=1)
    avg_filtered_model_err = np.mean(model_err * np.sqrt(model_weight), axis=1)

    fig = plt.figure(figsize=(16, 12), dpi=120)
    fig.suptitle("Model Error (Before Scaling)", fontsize=14)
    fig.subplots_adjust(hspace=0.5)
    base_subplot_value = 321
    plt.subplot(base_subplot_value)
    plt.title("Average")
    plt.plot(x_axis_range, avg_model_err, "r")
    for idx in state_indices.keys():
        base_subplot_value += 1
        plt.subplot(base_subplot_value)
        plt.title(f"{state_indices[idx]}")
        plt.plot(x_axis_range, model_err[:, idx], "r")

    # Set common labels
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.06, 0.5, "Error [m/s^2]", ha='center', va='center', rotation='vertical')

    if not show_plot:
        plt.savefig(os.path.join(results_dir, "fte_model_error.png"))
        plt.close()

    fig = plt.figure(figsize=(16, 12), dpi=120)
    fig.suptitle("Model Error (After Scaling)", fontsize=14)
    fig.subplots_adjust(hspace=0.5)
    base_subplot_value = 321
    plt.subplot(base_subplot_value)
    plt.title("Average")
    plt.plot(x_axis_range, avg_filtered_model_err, "g")
    for idx in state_indices.keys():
        base_subplot_value += 1
        plt.subplot(base_subplot_value)
        plt.title(f"{state_indices[idx]}")
        plt.plot(x_axis_range, model_err[:, idx] * np.sqrt(model_weight[idx]), "g")

    # Set common labels
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.06, 0.5, "Error [m/s^2]", ha='center', va='center', rotation='vertical')

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(results_dir, "fte_model_error_scaled.png"))
        plt.close()


def run_subset_tests(root_dir: str):
    fte_files = glob.glob(os.path.join(root_dir, "**/fte.pickle"), recursive=True)
    delta_acc_list = []
    for fte_file in fte_files:
        data = data_ops.load_pickle(fte_file)
        eval_dir = os.path.join(os.path.dirname(fte_file), "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        max_delta_acc = eval_delta_acc(data, eval_dir)
        eval_model_error(data, eval_dir)
        eval_meas_error(data, eval_dir)
        delta_acc_list.append(max_delta_acc)

    return np.array(delta_acc_list)


if __name__ == "__main__":
    parser = ArgumentParser(description="FTE Evaluation")
    parser.add_argument("--root_dir",
                        type=str,
                        help="The root directory where the reconstuction files (fte.pickle) is located")
    parser.add_argument("--type",
                        type=str,
                        help="Either process the runs with ('run'), 'flicks' with ('flick') or both with 'both'.")

    args = parser.parse_args()

    root_dir = os.path.normpath(args.root_dir)
    assert os.path.exists(root_dir), f"Data directory not found: {root_dir}"

    fte_files = glob.glob(os.path.join(root_dir, "**/fte.pickle"), recursive=True)
    delta_acc_list = []
    for fte_file in fte_files:
        assert os.path.isfile(fte_file), "fte.pickle file not found"
        if str(args.type) == "both" or str(args.type).lower() in fte_file.lower():
            print(f"{fte_file}", end=" ")
            data = data_ops.load_pickle(fte_file)

            eval_dir = os.path.join(os.path.dirname(fte_file), "evaluation")
            os.makedirs(eval_dir, exist_ok=True)

            # try:
            max_delta_acc = eval_delta_acc(data, eval_dir)
            eval_model_error(data, eval_dir)
            eval_meas_error(data, eval_dir)

            delta_acc_list.append(max_delta_acc)
            print("...done")
            # except:
            #     print("")
            #     print(f"Error: File did not have required data to process evaluation")

    # Calculate the average of the maximum acceleration differences i.e. max acc between time frames.
    delta_acc_list = np.array(delta_acc_list)
    avg_delta_acc = np.mean(delta_acc_list, axis=0)

    states = [
        "x_0",
        "y_0",
        "z_0",  # head position in inertial
        "phi_0",
        "theta_0",
        "psi_0",  # head rotation in inertial
        "phi_1",
        "theta_1",
        "psi_1",  # neck
        "theta_2",  # front torso
        "phi_3",
        "theta_3",
        "psi_3",  # back torso
        "theta_4",
        "psi_4",  # tail_base
        "theta_5",
        "psi_5",  # tail_mid
        "theta_6",
        "theta_7",  # l_shoulder, l_front_knee
        "theta_8",
        "theta_9",  # r_shoulder, r_front_knee
        "theta_10",
        "theta_11",  # l_hip, l_back_knee
        "theta_12",
        "theta_13",  # r_hip, r_back_knee
    ]
    # print(dict(zip(states, avg_delta_acc)))
