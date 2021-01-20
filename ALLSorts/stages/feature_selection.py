#=======================================================================================================================
#
#   ALLSorts v2 - Feature Selection Stage
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import message, _flat_hierarchy, _pseudo_counts

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from joblib import Parallel, delayed
from sklearn.base import clone
import pandas as pd
import numpy as np

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class FeatureSelection(BaseEstimator, TransformerMixin):
    '''Hierachical Mutual Information Feature Selection'''

    def __init__(self, cutoff=1, n_jobs=1, hierarchy=False, test=False, method="lr"):
        self.cutoff = cutoff
        self.hierarchy = hierarchy
        self.n_jobs = n_jobs
        self.method = method
        self.genes = {}
        self.test = test

    def _get_flat_hierarchy(self):
        return _flat_hierarchy(self.hierarchy, flat_hierarchy={})

    def _get_pseudo_counts(self, X, y, parents):
        return _pseudo_counts(X, y, parents, self.f_hierarchy)

    def _recurseFS(self, sub_hier, X, y, name="root"):

        if sub_hier == False:  # Recursion stop condition
            return False

        parents = list(sub_hier.keys())

        # Create pseudo-counts based on hiearachy
        X_p, y_p = self._get_pseudo_counts(X, y, parents)
        subtypes = list(y_p.unique())

        # Mutual information for each subtype on this level - combine
        if len(subtypes) > 2:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")( \
                delayed(self._feature_select)(X_p, y_p, subtype, multi=True) \
                        for subtype in subtypes)
            # Unpack results
            for sub_pos in range(len(subtypes)):
                self.genes[subtypes[sub_pos]] = results[sub_pos]

        else:
            subtype = "_".join(subtypes)
            shared_genes = self._feature_select(X_p, y_p, subtype, multi=False)
            self.genes[subtype] = shared_genes

        # Recurse through the hierarchy
        for parent in parents:
            self._recurseFS(sub_hier[parent],
                            X, y, name=parent)

    def _feature_select(self, X, y, subtype, multi=False):

        labels = y.copy()
        if multi:
            labels[~labels.isin([subtype])] = "Others"

        if self.method == "mi":

            fs_ = pd.Series(mutual_info_classif(X, labels), index=X.columns)
            genes = list(fs_[fs_ > self.cutoff * fs_.std()].sort_values(ascending=False).index)

        elif self.method == "lr":

            method = LogisticRegression(penalty="l1", C=1, solver="liblinear", class_weight="balanced")
            method.fit(X, labels)

            coefs = pd.DataFrame(method.coef_, columns=X.columns, index=["coef"])
            coefs = coefs[coefs != 0].dropna(axis=1)
            genes = list(coefs.columns)

        elif self.method == "all":
            genes = list(X.columns)

        elif self.test:
            genes = ["BCR", "ABL1", "HOXA6", "PBX1", "NUTM1", "HLF",
                     "RUNX1", "CRLF2", "DUX4", "MEF2C", "IAMP21_ratio", "CA6"]

        return genes

    def fit(self, X, y):
        self.f_hierarchy = self._get_flat_hierarchy()
        self._recurseFS(self.hierarchy, X, y)
        return self

    def transform(self, X, y=False):
        return {"genes": self.genes, "counts": X}

    def fit_transform(self, X, y=False):
        self.fit(X, y)
        return self.transform(X)





