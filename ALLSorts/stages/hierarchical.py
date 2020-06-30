#=======================================================================================================================
#
#   ALLSorts v2 - Hierarchical Classifier Stage
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import _flatHierarchy, _pseudoCounts, message

''' External '''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import pandas as pd
from sklearn.base import clone
from joblib import Parallel, delayed


''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_jobs=1, hierarchy=False, model=False, params=False):

        self.n_jobs = n_jobs
        self.hierarchy = hierarchy
        self.params = params
        self.model = model
        self.fitted = {}
        self.thresholds = {}
        self.trained = False

        # Choose model
        if not self.model:
            self.model = LogisticRegression(penalty="l1",
                                            solver="liblinear",
                                            class_weight="balanced")
        #if self.params:
        #    self.model.set_params(**params)

    def _flatHierarchy(self):
        return _flatHierarchy(self.hierarchy)

    def _pseudoCounts(self, X, y, name, parents):
        return _pseudoCounts(X, y, name, parents, self.f_hierarchy)

    def _setThresholds(self):
        for subtype in self.f_hierarchy:
            self.thresholds[subtype] = 0.5

    def _getParents(self):
        parents = []
        for subtype in self.f_hierarchy:
            if self.f_hierarchy[subtype]:
                parents.append(subtype)

        return parents

    def _checkInput(self, X):
        # Prepare
        if isinstance(X, dict):
            counts = X["counts"]
            self.genes = X["genes"]
        else:
            counts = X.copy()
            self.genes = False

        return counts

    def _clf(self, X, y, model, subtype, multi=False):

        # Prepare new labels per this subtype
        train_y = y.copy()
        if multi:
            train_y[~train_y.isin([subtype])] = "Others"

        # Refine genes if custom feature selection
        counts = X[self.genes[subtype]] if self.genes else X.copy()
        fitted = model.fit(counts, train_y)

        return fitted

    def _recurseCLF(self, sub_hier, X, y, name="root"):

        if sub_hier == False:  # Recursion stop condition
            return False

        parents = list(sub_hier.keys())

        # Create pseudo-counts based on hiearachy
        X_p, y_p = self._pseudoCounts(X, y, name, parents)
        subtypes = list(y_p.unique())

        # OVR if multiple subtypes, single binary if 2.
        if len(subtypes) > 2:
            results = Parallel(n_jobs=1, prefer="threads")( \
                delayed(self._clf)(X_p, y_p, clone(self.model), subtype, multi=True) \
                for subtype in subtypes)

            # Unpack results
            for sub_pos in range(len(subtypes)):
                self.fitted[subtypes[sub_pos]] = results[sub_pos]

        else:
            subtype = "_".join(subtypes)
            fitted = self._clf(X_p, y_p, clone(self.model), subtype, multi=False)
            self.fitted[subtype] = fitted

        # Recurse through the hierarchy
        for parent in parents:
            self._recurseCLF(sub_hier[parent],
                             X, y, name=parent)

    def _weightedProba(self, probs):
        probs_weighted = probs.copy()

        for parent in self.f_hierarchy:
            if self.f_hierarchy[parent] != False:  # has children
                for child in self.f_hierarchy[parent]:
                    preds = probs_weighted[child].multiply(probs_weighted[parent], axis="index")
                    probs_weighted[child] = list(preds)

        return probs_weighted

    def _evalPreds(self, predicted_proba, p=False):

        y_pred = []

        # Define parent/child relationships
        children = []
        parents = []
        for parent in self.f_hierarchy:
            if self.f_hierarchy[parent] != False:
                parents.append(parent)
                children.append(self.f_hierarchy[parent])

        # Check each result
        for sample, probabilities in predicted_proba.iterrows():

            # If unclassified, call it from the start
            if probabilities["Pred"] == "":
                y_pred.append("Unclassified")
                continue

            # Otherwise, unpack results
            prediction = probabilities["Pred"][:-1]  # Remove final comma
            predictions = prediction.split(",")

            # Cull if parent node is not successful
            for i in range(0, len(children)):
                pred_children = list(set(predictions).intersection(set(children[i])))
                has_children = len(pred_children) > 0

                if p:
                    if has_children and parents[i] not in predictions:
                        predictions = list(set(predictions).difference(set(children[i])))
                    elif has_children and parents[i] in predictions:
                        predictions.remove(parents[i])

                # Groups at the moment are mutually exclusive
                if len(pred_children) > 1 and len(predictions) > 1:
                    prob_no = probabilities[pred_children].apply(pd.to_numeric)
                    max_subtype = [prob_no.idxmax()]
                    drop_preds = list(set(prob_no.index.values).difference(set(max_subtype)))
                    for drop_pred in drop_preds:
                        try:
                            predictions.remove(drop_pred)
                        except ValueError:
                            continue

            # Unclassified
            if len(predictions) == 0:
                y_pred.append("Unclassified")

            # Multi-label
            elif len(predictions) > 1:
                y_pred.append(','.join(predictions))

            # Single prediction
            else:
                y_pred.append(predictions[0])

        return y_pred


    def _filterHealthy(self, probabilities, counts):

        probabilities.loc[(counts["B-ALL"] < -3) | (counts["B-ALL"] > 3)] = 0.0
        return probabilities


    def predict_proba(self, X, parents=False, filter_healthy=False):

        '''
        Retrieve the unweighted probabilities

           Input: Pre-processed counts matrix
           Output: DataFrame of probabilities
        '''

        counts = self._checkInput(X)
        predicted_proba = pd.DataFrame(index=counts.index)

        for subtype in self.fitted:

            probs = self.fitted[subtype].predict_proba(counts[self.genes[subtype]])
            probs = pd.DataFrame(probs,
                                 columns=self.fitted[subtype].classes_,
                                 index=counts.index)

            if "Others" in probs.columns:
                probs.drop(["Others"], axis=1, inplace=True)

            predicted_proba = pd.concat([predicted_proba, probs], axis=1, join="inner")

        predicted_proba = self._weightedProba(predicted_proba)
        if not parents:
            predicted_proba.drop(self._getParents(), inplace=True, axis=1)


        predicted_proba = self._filterHealthy(predicted_proba, counts)
        return predicted_proba

    def predict(self, X, parents=True):

        predicted_proba = self.predict_proba(X, parents=parents)
        predicted_proba["Pred"] = ""

        for subtype, probabilities in predicted_proba.drop("Pred", axis=1).iteritems():
            predicted_proba.loc[probabilities > self.thresholds[subtype], "Pred"] += subtype + ","

        y_pred = self._evalPreds(predicted_proba, p=parents)

        return y_pred

    def fit(self, X, y):

        counts = self._checkInput(X)
        self.f_hierarchy = self._flatHierarchy()
        self._setThresholds()

        # Train hierarchical classifier
        self._recurseCLF(self.hierarchy, counts, y)

        return self

