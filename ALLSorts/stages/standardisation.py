#=======================================================================================================================
#
#   ALLSorts v2 - Standardisation Stage
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import message

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Transformers
from sklearn.preprocessing import StandardScaler, PowerTransformer

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class Scaler(BaseEstimator, TransformerMixin):

    def __init__(self, scaler="std"):
        self.options = {"std": StandardScaler(), "pwr": PowerTransformer()}
        self.scaler = scaler

    def _checkInput(self, X):
        # Prepare
        if isinstance(X, dict):
            counts = X["counts"]
            self.genes = X["genes"]
        else:
            counts = X.copy()
            self.genes = False

        return counts

    def fit(self, X, y=False):
        self.scaled = self.options[self.scaler]
        counts = self._checkInput(X)
        self.scaled = self.scaled.fit(counts, y)
        return self

    def transform(self, X, y=False):
        counts = self._checkInput(X)

        scaled = self.scaled.transform(counts)
        scaled = pd.DataFrame(scaled, columns=counts.columns, index=counts.index)

        return {"genes": self.genes, "counts": scaled}

    def fit_transform(self, X, y=False):
        self.fit(X)
        return self.transform(X)