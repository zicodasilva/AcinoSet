import os
from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd
from glob import glob
from lib import misc, utils, app, metric
from lib.calib import triangulate_points_fisheye, project_points_fisheye
from py_utils import data_ops, log

import matplotlib.pyplot as plt

plt.style.use(os.path.join("../configs", "mechatronics_style.yaml"))
mechatronics_orange = '#FF6400'
mechatronics_charcoal = '#5A5A5A'

# Data gathered for the number of measurements included for the paws and ankles.
# g_meas_stats = np.array([[39.27, 44.4, 24.59, 22.61, 22.44, 27.56, 22.61, 21.29],
#                          [31.47, 24.33, 30.95, 39.95, 29.81, 25.36, 33.64, 23.08],
#                          [38.39, 38.21, 50.43, 50.99, 38.89, 40, 42.47, 41.36],
#                          [31.57, 27.56, 33.88, 33.55, 31.96, 29.37, 32.84, 24.75],
#                          [46.67, 43.25, 57.92, 58.67, 35.42, 28.42, 44.33, 36.33]])


def generate_histograms(acinoset_fname: str, pw_fname: str):
    acinoset = data_ops.load_pickle(acinoset_fname)
    pw = data_ops.load_pickle(pw_fname)

    # plot the error histogram
    xlabel = "Error [px]"
    ylabel = "Density"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(acinoset["error"], bins=50, density=True, color=mechatronics_orange, alpha=0.5)
    ax.hist(pw["error"], bins=50, density=True, color=mechatronics_charcoal, alpha=0.5)
    # ax.set_title("Error Distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 60])
    labels = ["FTE", "PW-FTE"]
    ax.legend(labels)
    fig.savefig(os.path.join(os.path.dirname(acinoset_fname), "error_distribution.png"))


if __name__ == "__main__":
    generate_histograms("/Users/zico/msc/dev/AcinoSet/data/2019_03_09/jules/flick2/fte_pw/reprojection.pickle",
                        "/Users/zico/msc/dev/AcinoSet/data/2019_03_09/jules/flick2/fte_pw_pw/reprojection.pickle")
