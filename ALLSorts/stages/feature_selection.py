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
from ALLSorts.common import message, _flatHierarchy, _pseudoCounts

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin

# Methods
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

    def __init__(self, cutoff=2, n_jobs=1, hierarchy=False, test=False, method=mutual_info_classif):
        self.cutoff = cutoff
        self.hierarchy = hierarchy
        self.n_jobs = n_jobs
        self.method = method
        self.genes = {}
        self.test = test

    def _flatHierarchy(self):
        return _flatHierarchy(self.hierarchy)

    def _pseudoCounts(self, X, y, name, parents):
        return _pseudoCounts(X, y, name, parents, self.f_hierarchy)

    def _recurseFS(self, sub_hier, X, y, name="root"):

        if sub_hier == False:  # Recursion stop condition
            return False

        parents = list(sub_hier.keys())

        # Create pseudo-counts based on hiearachy
        X_p, y_p = self._pseudoCounts(X, y, name, parents)
        subtypes = list(y_p.unique())

        # Mutual information for each subtype on this level - combine
        if len(subtypes) > 2:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")( \
                delayed(self._feature_select)(X_p, y_p, subtype, self.method, multi=True) \
                        for subtype in subtypes)

            # Unpack results
            for sub_pos in range(len(subtypes)):
                self.genes[subtypes[sub_pos]] = results[sub_pos]

        else:
            subtype = "_".join(subtypes)
            shared_genes = self._feature_select(X_p, y_p, subtype, self.method, multi=False)
            self.genes[subtype] = shared_genes

            # Recurse through the hierarchy
        for parent in parents:
            self._recurseFS(sub_hier[parent],
                            X, y, name=parent)

    def _feature_select(self, X, y, subtype, method, multi=False):

        labels = y.copy()
        if multi:
            labels[~labels.isin([subtype])] = "Others"

        if self.cutoff == 0 and not self.test:
            genes = list(X.columns)
        elif not self.test:
            fs = method(X, labels)
            fs_ = pd.Series(fs, index=X.columns)
            genes = list(fs_[fs_ > self.cutoff * fs_.std()].sort_values(ascending=False).index)
        else:
            genes = ["BCR", "ABL1", "HOXA6", "PBX1", "NUTM1", "HLF",
                     "RUNX1", "CRLF2", "DUX4", "MEF2C", "IAMP21_ratio", "CA6"]

        return genes

    def fit(self, X, y):
        self.f_hierarchy = self._flatHierarchy()
        self._recurseFS(self.hierarchy, X, y)
        return self

    def transform(self, X, y=False):
        return {"genes": self.genes, "counts": X}

    def fit_transform(self, X, y=False):
        self.fit(X, y)
        return self.transform(X)





