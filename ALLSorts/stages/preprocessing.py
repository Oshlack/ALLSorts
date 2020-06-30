#=======================================================================================================================
#
#   ALLSorts v2 - Pre-processing Stage
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
from morp import morp
import numpy as np


''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class Preprocessing(BaseEstimator, TransformerMixin):

    ''' Input: Raw counts
        Output: Processed counts for input into ALLSorts

        Method: Filter > Normalise > Log > Scale
    '''

    def __init__(self,
                 filter=True, min_indiv=20, min_total=100,
                 norm="MOR",
                 log=True,
                 truncate=True):

        self.filter = filter
        self.min_indiv = min_indiv
        self.min_total = min_total
        self.truncate = truncate
        self.log = log
        self.norm = norm

    def _filter(self, counts):

        ''' Filter by: Max count in any sample for a gene
                       Sum of all counts per gene
        '''

        return counts.loc[:, (counts.max(axis=0) >= self.min_indiv) &
                             (counts.sum(axis=0) >= self.min_total)]

    def _truncate(self, counts, transform=False):

        ''' Truncate maximum values of genes '''

        if not transform:
            self.ceiling = (counts.median() +
                            3 * (counts.quantile(0.75) -
                                 counts.quantile(0.25)))

        return counts.clip(upper=self.ceiling, axis=1)

    def _normalise(self, counts, norm="MOR"):

        ''' Normalise by: TMM: Trimmed Mean of M-values
                          MOR: Median of ratios method
        '''

        if norm == "MOR":
            self.normalise = morp.MORP(counts.transpose())

        return self.normalise.mor.transpose()

    def fit(self, counts, y=False):

        # Filter
        if self.filter:
            counts = self._filter(counts)

        # Normalise
        if self.norm:
            counts = self._normalise(counts, self.norm)

        # Truncate
        if self.truncate:
            counts = self._truncate(counts)

        # Transform
        if self.log:
            counts = np.log2(counts + 1)

        self.genes = list(counts.columns)

        return self

    def transform(self, counts, y=False):

        # Filter genes
        if self.filter:
            counts = counts.reindex(self.genes, axis=1).dropna(axis=1)

        # Normalise
        if self.norm:
            counts = self.normalise.transformMor(counts.transpose()).transpose()

        # Truncate extreme values
        if self.truncate:
            counts = self._truncate(counts, transform=True)

        # Log
        if self.log:
            counts = np.log2(counts + 1)

        # Processed
        return counts

    def fit_transform(self, counts, y=False):
        self.fit(counts)

        return self.transform(counts)