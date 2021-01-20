#=======================================================================================================================
#
#   ALLSorts v2 - Rebuild comparisons for visualisations
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import root_dir

''' External '''
import numpy as np
import pandas as pd
import joblib
from umap import UMAP

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def rebuild_comparisons(allsorts_clf, probabilities, ui, size=10):

    X_filtered = allsorts_clf.transform(ui.samples)["counts"]

    ''' Probability comparisons '''

    true_samples = []
    for sample, column in probabilities.iterrows():
        if column["True"] in column["Pred"]:
            true_samples.append(sample)

    comparisons = probabilities.loc[true_samples]
    cgroup = comparisons.groupby("True")
    new_samples = cgroup.apply(lambda x: x.sample(n=size)
                                if x.shape[0] > size
                                else x.sample(n=x.shape[0]))

    new_index = []
    for sample in new_samples.index:
        new_index.append(sample[1])

    filename = "comparisons.csv"
    destination = str(root_dir())+"/models/allsorts/" + filename
    cfinal = comparisons[(comparisons.index.isin(new_index))]
    cfinal.to_csv(destination)

    '''Umap Visualisation'''

    # Get genes used in the final models and filter counts by them
    chosen_genes = []
    classifiers = allsorts_clf.steps[-1][-1].fitted
    feature_selection = allsorts_clf.steps[-3][-1].genes

    for subtype, clf in classifiers.items():
        genes = feature_selection[subtype]
        sub_genes = pd.DataFrame(clf.coef_, columns=genes)
        sub_genes = sub_genes[sub_genes != 0].dropna(axis=1)
        chosen_genes += list(sub_genes.columns)

    chosen_genes = list(set(chosen_genes))
    X_filtered = X_filtered.loc[true_samples, chosen_genes]
    labels = ui.labels.loc[true_samples]

    # Create UMAPs
    u = UMAP(n_neighbors=10).fit(X_filtered)

    # Save
    u_filename = 'umap.sav'
    l_filename = 'comparison_labels.csv'
    g_filename = 'comparison_genes.csv'

    destination = str(root_dir()) + "/models/allsorts/comparisons/" + u_filename
    joblib.dump(u, destination)
    labels.to_csv(str(root_dir()) + "/models/allsorts/comparisons/" + l_filename)
    pd.Series(X_filtered.columns).to_csv(str(root_dir()) + "/models/allsorts/comparisons/" + g_filename)






