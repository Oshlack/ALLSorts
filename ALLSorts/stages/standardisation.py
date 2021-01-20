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

    def _check_input(self, X):
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
        counts = self._check_input(X)
        self.scaled = self.scaled.fit(counts, y)
        return self

    def transform(self, X, y=False):

        counts = self._check_input(X)

        scaled = self.scaled.transform(counts)
        scaled = pd.DataFrame(scaled, columns=counts.columns, index=counts.index)
        scaled = scaled.fillna(0.0)

        return scaled

