from average_tf_map import average_tf
import numpy as np
import os
import scipy.stats as st
from mne.stats import fdr_correction


def stats(arr1):

    p_all = np.zeros((arr1.shape[1], arr1.shape[-2], arr1.shape[-1]))
    for ch in range(arr1.shape[1]):

        # test nieparametryczny
        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.stats.wilcoxon.html
        for f in range(arr1.shape[-2]):
            for t in range(arr1.shape[-1]):
                z, p_all[ch, f, t] = st.wilcoxon(arr1[:, ch, 0, f, t].flatten(),
                                                 arr1[:, ch, 1, f, t].flatten())

                p_all[ch, f, t] = p_all[ch, f, t] / 2

        # kontrola frakcji fałszywych odkryć (FDR)
        # https://mne.tools/dev/auto_examples/stats/fdr_stats_evoked.html#sphx-glr-auto-examples-stats-fdr-stats-evoked-py
    reject_fdr, pval_fdr = fdr_correction(p_all, alpha=0.05, method='indep')  # można macierz

    return p_all, reject_fdr


if __name__ == "__main__":
    filename = os.path.join("output", "data_matrix.npy")
    average = average_tf(filename)  # 19 kan, 2 serie, 60 x 500
    print(average.shape)
    stats(average)
