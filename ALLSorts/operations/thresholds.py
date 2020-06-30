#=======================================================================================================================
#
#   ALLSorts v2 - Calculate optimal thresholds
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import message, root_dir

''' External '''
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
import numpy as np
import pandas as pd

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def _score_thresholds(probs, y, subtype):

    # Calculate Threshold
    auc = roc_auc_score(list(y), probs)

    if auc != 1 and auc != 0:  # For binary case
        precision, recall, thresh = precision_recall_curve(list(y), probs)

        f1 = 2 * ((precision * recall) / (precision + recall))
        threshold = thresh[np.argmax(f1)]
    else:
        prob_dist = pd.concat([probs, y], axis=1, join="inner")
        prob_dist.columns = ["Prob", "Label"]
        threshold = (prob_dist[prob_dist["Label"] == 1].min()["Prob"] +
                     prob_dist[prob_dist["Label"] != 1].max()["Prob"]) / 2

    return float(threshold)


def fit_thresholds(probabilities, f_hierarchy, y):

    thresholds = {}

    for subtype in f_hierarchy.keys():

        select = [subtype] if not f_hierarchy[subtype] else f_hierarchy[subtype]

        labels = y.copy()
        labels[labels.isin(select)] = 1
        labels[labels != 1] = 0
        threshold = _score_thresholds(probabilities[subtype], labels, subtype)

        if threshold > 0.8:
            threshold = 0.8
        elif threshold < 0.2:
            threshold = 0.2

        thresholds[subtype] = threshold

    return thresholds
