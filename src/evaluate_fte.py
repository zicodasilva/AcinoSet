import os
import glob
from typing import Dict
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from py_utils import data_ops

g_states = [
    "x_0", "y_0", "z_0",         # head position in inertial
    "phi_0", "theta_0", "psi_0", # head rotation in inertial
    "phi_1", "theta_1", "psi_1", # neck
    "theta_2",                   # front torso
    "phi_3", "theta_3", "psi_3", # back torso
    "theta_4", "psi_4",          # tail_base
    "theta_5", "psi_5",          # tail_mid
    "theta_6", "theta_7",        # l_shoulder, l_front_knee
    "theta_8", "theta_9",        # r_shoulder, r_front_knee
    "theta_10", "theta_11",      # l_hip, l_back_knee
    "theta_12", "theta_13",      # r_hip, r_back_knee
]

def eval_delta_acc(data: Dict, results_dir: str) -> None:
    start_frame = data["start_frame"]
    acc = np.array(data["ddx"])
    x_axis_range = range(start_frame, start_frame + len(acc))
    delta_acc = np.array([(acc[n, :] - acc[n-1, :]) for n in range(1, len(acc))])

    fig = plt.figure()
    fig.suptitle("Acceleration and Delta Acceleration (-) States", fontsize=14)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.subplot(221)
    plt.plot(x_axis_range, acc[:, 17], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 17], "--b")
    plt.title("Left Shoulder")

    plt.subplot(222)
    plt.plot(x_axis_range, acc[:, 18], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 18], "--b")
    plt.title("Left Front Knee")

    plt.subplot(223)
    plt.plot(x_axis_range, acc[:, 23], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 23], "--b")
    plt.title("Right Hip")

    plt.subplot(224)
    plt.plot(x_axis_range, acc[:, 24], "g")
    plt.plot(x_axis_range[:-1], delta_acc[:, 24], "--b")
    plt.title("Right Back Knee")

    # Set common labels
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.04, 0.5, "Acceleration [m/s^2]", ha='center', va='center', rotation='vertical')

    plt.savefig(os.path.join(results_dir, "fte_acceleration.png"))
    plt.close()

    data_ops.save_pickle(os.path.join(results_dir, "delta_acc.pickle"), dict(zip(g_states, np.max(np.abs(delta_acc), axis=0))))

def eval_meas_error(data: Dict, results_dir: str) -> None:
    start_frame = data["start_frame"]
    meas_err = data["meas_err"]
    meas_weight = data["meas_weight"]

    x_axis_range = range(start_frame, start_frame + len(meas_weight))

    meas_weight = np.expand_dims(meas_weight, 3)
    weighted_meas_err = meas_weight * meas_err
    avg_meas_err = np.array([np.mean(meas_err[:, cam_idx], axis=(1, 2)) for cam_idx in range(6)])
    avg_filtered_meas_err = np.array([np.mean(weighted_meas_err[:, cam_idx], axis=(1, 2)) for cam_idx in range(6)])

    fig = plt.figure()
    fig.suptitle("Reprojection Error (Before Filtering and Scaling)", fontsize=14)
    fig.subplots_adjust(hspace=0.5)
    base_subplot_value = 320
    for idx in range(6):
        base_subplot_value += 1
        plt.subplot(base_subplot_value)
        plt.title(f"CAM {idx+1}")
        plt.plot(x_axis_range, avg_meas_err[idx, :], "r")

    # Set common labels
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.06, 0.5, "Error [pixels]", ha='center', va='center', rotation='vertical')

    plt.savefig(os.path.join(results_dir, "fte_meas_error.png"))
    plt.close()

    fig = plt.figure()
    fig.suptitle("Reprojection Error (After Filtering and Scaling)", fontsize=14)
    fig.subplots_adjust(hspace=0.5)
    base_subplot_value = 320
    for idx in range(6):
        base_subplot_value += 1
        plt.subplot(base_subplot_value)
        plt.title(f"CAM {idx+1}")
        plt.plot(x_axis_range, avg_filtered_meas_err[idx, :], "g")

    # Set common labels
    fig.text(0.5, 0.04, "Frame Number", ha='center', va='center')
    fig.text(0.06, 0.5, "Error [pixels]", ha='center', va='center', rotation='vertical')

    plt.savefig(os.path.join(results_dir, "fte_meas_error_filtered.png"))
    plt.close()

def eval_model_error(data: Dict, results_dir: str) -> None:
    start_frame = data["start_frame"]
    model_err = data["model_err"]
    model_weight = data["model_weight"]

    x_axis_range = range(start_frame, start_frame + len(model_err))
    avg_model_err = np.mean(model_err[:, model_weight != 0], axis=1)
    avg_filtered_model_err = np.mean(model_err * np.sqrt(model_weight), axis=1)

    plt.figure()
    plt.subplot(211)
    plt.title("Average Model Error")
    plt.plot(x_axis_range, avg_model_err, "b", label="Before Scaling")
    plt.legend(loc="upper right", fontsize="xx-small")
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)

    plt.subplot(212)
    plt.plot(x_axis_range, avg_filtered_model_err, "b", label="After Scaling")
    plt.xlabel("Frame Number", fontsize=10)
    plt.legend(loc="upper right", fontsize="xx-small")

    plt.savefig(os.path.join(results_dir, "fte_model_error.png"))
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="FTE Evaluation")
    parser.add_argument("--data_dir", type=str, help="The directory where the reconstuction file (fte.pickle) is located")

    args = parser.parse_args()

    data_dir = os.path.normpath(args.data_dir)
    assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"
    file = os.path.join(data_dir, "fte.pickle")
    assert os.path.isfile(file), "fte.pickle file not found"

    eval_dir = os.path.join(data_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    data = data_ops.load_pickle(file)

    eval_delta_acc(data, eval_dir)
    eval_model_error(data, eval_dir)
    eval_meas_error(data, eval_dir)

    acc_files = glob.glob(os.path.join("/Users/zico/msc/dev/AcinoSet/data", "**/delta_acc.pickle"), recursive=True)
    delta_acc_list = []
    for f_acc in acc_files:
        temp = data_ops.load_pickle(f_acc)
        delta_acc_list.append(list(temp.values()))

    delta_acc_list = np.array(delta_acc_list)
    avg_delta_acc = np.mean(delta_acc_list, axis=0)

    print(dict(zip(g_states, avg_delta_acc)))

