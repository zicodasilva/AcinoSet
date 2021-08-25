import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import main_fte

plt.style.use(os.path.join("../configs", "mplstyle.yaml"))

if __name__ == "__main__":
    root_dir = os.path.join(
        "/Users/zico/OneDrive - University of Cape Town/CheetahReconstructionResults/cheetah_videos")
    root_results_dir = "/Users/zico/msc/data/PairwiseExperimentResults3"
    burst_lengths = (1, 5, 10, 15)
    num_drop_outs = (0, 25, 50, 75, 90)
    drop_out_range = (20, 100)
    data_path = os.path.join("2017_08_29", "top", "jules", "run1_1")
    start_frame = 10
    end_frame = 110
    dlc_thresh = 0.5

    tests = ("Normal", "Pairwise")
    filtered_markers = ("r_front_ankle", "r_front_paw", "r_back_ankle", "r_back_paw")
    drop_out_dataset = pd.read_csv(os.path.join(root_results_dir, "manual_drop_outs.csv"))
    for test in tests:
        for burst in burst_lengths:
            for num_filtered in num_drop_outs:
                out_prefix = os.path.join(root_results_dir, test, f"{num_filtered}_percent_{burst}_burst")
                try:
                    drop_out_frames = drop_out_dataset[str((burst, num_filtered))].values
                except KeyError:
                    continue
                print(f"Run test: {out_prefix}")
                drop_out_frames = drop_out_frames[~np.isnan(drop_out_frames)]
                drop_out_frames = drop_out_frames.astype(int).tolist()
                # Run the optimisation
                main_fte.run(root_dir,
                             data_path,
                             start_frame,
                             end_frame,
                             dlc_thresh,
                             filtered_markers=filtered_markers,
                             drop_out_frames=drop_out_frames,
                             pairwise_included=2 if test == "Pairwise" else 0,
                             out_dir_prefix=out_prefix)
                # Produce results
                _, _ = main_fte.metrics(root_dir,
                                        data_path,
                                        start_frame,
                                        end_frame,
                                        dlc_thresh,
                                        out_dir_prefix=out_prefix)
