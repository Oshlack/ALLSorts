#=======================================================================================================================
#
#   ALLSorts v2 - Find Centroids
#   Not all subtypes are destined to be classified by a small set of genes, they are defined by group membership.
#
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

# Methods
import pandas as pd

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class CentroidCreate(BaseEstimator, TransformerMixin):

	"""
	A class that represents the custom feature that represents the distance to each subtype centroid.

	...

	Attributes
	__________
	BaseEstimator : inherit
		Inherit from Scikit-learn BaseEstimator class
	TransformerMixin : inherit
		Inherit from Scikit-learn TransformerMixin class

	Methods
	-------
	fit(counts, y)
		Get all centroid parameters relative to the training set.
	transform(counts)
		Transform input counts by parameters determined by training set.

	"""

	def __init__(self, hierarchy=False, distance_function=False, only=False):

		"""
		Initialise the class

		Parameters
		__________
		hierarchy : dict
			Dictionary representing the hierarchy of subtypes
		distance_function : object
			The distance function to calculate between centroids and samples. i.e. cosine_similarity.
		only : bool
			Discard all other features, keep only the distances to centroids.
		"""

		self.hierarchy = hierarchy
		self.cv_genes = {}
		self.distance_function = distance_function
		self.centroids = {}
		self.kpca = {}
		self.only = only
		self.scaler = {}
		self.training = True

	def _check_input(self, X):
		# Prepare
		if isinstance(X, dict):
			counts = X["counts"]
			self.genes = X["genes"]
		else:
			counts = X.copy()
			self.genes = False

		return counts

	def _get_flat_hierarchy(self):
		return _flat_hierarchy(self.hierarchy, flat_hierarchy={})

	def _get_pseudo_counts(self, X, y, parents):
		return _pseudo_counts(X, y, parents, self.f_hierarchy)

	def _recurse_subtypes(self, sub_hier, X, y, name="root"):

		if sub_hier == False:  # Recursion stop condition
			return False

		parents = list(sub_hier.keys())

		# Create pseudo-counts based on hiearachy
		X_p, y_p = self._get_pseudo_counts(X, y, parents)
		subtypes = list(y_p.unique())

		if len(subtypes) > 2:
			for subtype in subtypes:

				labels = y_p.copy()
				labels[~labels.isin([subtype])] = "Others"

				rf = RandomForestClassifier(n_estimators=100, class_weight="balanced").fit(X_p, labels)
				coefs = pd.DataFrame(rf.feature_importances_, index=X_p.columns)[0]
				self.cv_genes[subtype] = list(coefs.abs().sort_values(ascending=False).iloc[0:20].index)
				self.kpca[subtype] = KernelPCA(2, kernel="rbf").fit(X_p[self.cv_genes[subtype]])

				X_p_pca = pd.DataFrame(self.kpca[subtype].transform(X_p[self.cv_genes[subtype]]), index=X_p.index)

				# Now calculate centroids
				centroid = X_p_pca.loc[labels == subtype].median()
				self.centroids[subtype] = pd.DataFrame(centroid).transpose()

		# Recurse through the hierarchy
		for parent in parents:
			self._recurse_subtypes(sub_hier[parent], X, y, name=parent)

	def _centroid_similarity(self, counts, y):

		distance_features = pd.DataFrame(index=counts.index)

		for subtype in self.centroids.keys():

			genes = self.cv_genes[subtype]
			counts_pca = pd.DataFrame(self.kpca[subtype].transform(counts[genes]), index=counts.index)
			distance = self.distance_function(counts_pca, self.centroids[subtype])

			distance_ = []
			for d in distance:
				distance_.append(d[0])

			distance_features["distance_"+subtype] = distance_

		if self.training:
			self.scaler = StandardScaler().fit(distance_features)

		distance_features = pd.DataFrame(self.scaler.transform(distance_features),
										 columns = distance_features.columns,
										 index = distance_features.index)

		counts = pd.concat([counts, distance_features], axis=1, join="inner")

		if self.training:
			for subtype in self.centroids.keys():

				if self.only:
					self.genes[subtype] = list(distance_features.columns)
				else:
					self.genes[subtype] += list(distance_features.columns)


			self.training = False

		return counts

	def fit(self, X, y):

		counts = self._check_input(X)

		self.f_hierarchy = self._get_flat_hierarchy()
		self._recurse_subtypes(self.hierarchy, counts, y)

		return self

	def transform(self, X, y=False):

		counts = self._check_input(X)
		counts = self._centroid_similarity(counts, y)

		return {"genes": self.genes, "counts": counts}





