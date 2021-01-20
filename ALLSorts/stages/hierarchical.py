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
from ALLSorts.common import _flat_hierarchy, _pseudo_counts, message

''' External '''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.base import clone
from joblib import Parallel, delayed

import numpy as np

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):

	"""
	A class that represents a Hierarchical Classifier.

	A hierarchical classifier in this context is simply a One vs. Rest classifier at each parent node in a hierarchy -
	supplied as a dictionary, representing a tree.

	...

	Attributes
	__________
	BaseEstimator : Scikit-Learn BaseEstimator class
		Inherit from this class.
	ClassifierMixin : Scikit-Learn ClassifierMixin class
		Inherit from this class.

	Methods
	-------
	fit(X, y)
		With supplied training counts and labels, as transformed through the ALLSorts pipeline, fit/train the
		hierarchical classifier.
		Note: These should be distinct from any samples you wish to validate the result with.
	predict(X, parents=False)
		With the supplied counts, transformed through the ALLSorts pipeline, return a list of predictions.
	predict_proba(X, parents=False)
		With the supplied counts, transformed through the ALLSorts pipeline, return a list of probabilities and
		predictions.

	"""

	def __init__(self, n_jobs=1, hierarchy=False, model=False, params=False):

		"""
		Initialise the class

		Attributes
		__________
		n_jobs : int
			How many threads to use to train the classifier(s) concurrently.
		hierarchy : dict
			A representation of the hierarhical tree.
		model : object
			A scikit learn classifier, i.e. LogisticRegression(*).
		params : dict
			Parameters to use within the model provided.

		"""

		self.n_jobs = n_jobs
		self.hierarchy = hierarchy
		self.params = params
		self.model = model
		self.fitted = {}
		self.thresholds = {}
		self.filter_healthy = False
		self.trained = False

		if not self.model:
			self.model = LogisticRegression(penalty="l1",
											solver="liblinear",
											class_weight="balanced")
		if self.params:
			self.model.set_params(**params)

	def _get_flat_hierarchy(self):
		return _flat_hierarchy(self.hierarchy, flat_hierarchy={})

	def _get_pseudo_counts(self, X, y, parents):
		return _pseudo_counts(X, y, parents, self.f_hierarchy)

	def _set_thresholds(self):

		""" Set default thresholds, this will be overridden eventually during cross-validation. """
		for subtype in self.f_hierarchy:
			self.thresholds[subtype] = 0.5

	def _get_parents(self):
		parents = []
		for subtype in self.f_hierarchy:
			if self.f_hierarchy[subtype]:
				parents.append(subtype)

		return parents

	def _check_input(self, X):

		if isinstance(X, dict):
			counts = X["counts"]
			self.genes = X["genes"]
		else:
			counts = X.copy()
			self.genes = False

		return counts

	def _clf(self, X, y, model, subtype, multi=False):

		train_y = y.copy()
		if multi:
			train_y[~train_y.isin([subtype])] = "Others"

		''' Refine genes if custom feature selection '''
		counts = X[self.genes[subtype]] if self.genes else X.copy()
		fitted = model.fit(counts, train_y)

		''' How many genes in this subtype '''
		coefs = pd.DataFrame(fitted.coef_, columns=counts.columns).transpose()
		coefs.columns = ["coef"]
		genes = coefs[coefs != 0].dropna().abs().sort_values(ascending=False, by="coef")

		'''
		print("\n")
		print(subtype, genes.shape)
		print(list(genes.index))
		'''

		return fitted

	def _recurse_clf(self, sub_hier, X, y):

		if sub_hier == False:  # Recursion stop condition
			return False

		parents = list(sub_hier.keys())
		X_p, y_p = self._get_pseudo_counts(X, y, parents)
		subtypes = list(y_p.unique())

		'''OVR if multiple subtypes, single binary if two subtypes '''
		if len(subtypes) > 2:
			results = Parallel(n_jobs=1, prefer="threads")(
								delayed(self._clf)(X_p, y_p, clone(self.model), subtype, multi=True)
								for subtype in subtypes)

			for sub_pos in range(len(subtypes)):
				self.fitted[subtypes[sub_pos]] = results[sub_pos]

		else:
			subtype = "_".join(subtypes)
			fitted = self._clf(X_p, y_p, clone(self.model), subtype, multi=False)
			self.fitted[subtype] = fitted

		''' Recurse through the hierarchy '''
		for parent in parents:
			self._recurse_clf(sub_hier[parent], X, y)

	def _weighted_proba(self, probs):

		""" Weight the probabilities of subtypes by multiplying by the parent, meta-subtype."""

		probs_weighted = probs.copy()

		for parent in self.f_hierarchy:
			if self.f_hierarchy[parent] != False:  # has children
				for child in self.f_hierarchy[parent]:
					preds = probs_weighted[child].multiply(probs_weighted[parent], axis="index")
					probs_weighted[child] = list(preds)

		return probs_weighted

	def _parent_children(self):
		children = []
		parents = []
		for parent in self.f_hierarchy:
			if self.f_hierarchy[parent] != False:
				parents.append(parent)
				children.append(self.f_hierarchy[parent])

		return children, parents

	def _eval_preds(self, predicted_proba, p=False):

		""" This is a bit of a messy function. Essentially, given the predictions, it attempts to remove
			redundancies. For example, if a child node is classified but the meta-subtype is not. """

		y_pred = []
		children, parents = self._parent_children()

		''' Check each result '''
		for sample, probabilities in predicted_proba.iterrows():

			if probabilities["Pred"] == "":
				y_pred.append("Unclassified")
				continue

			prediction = probabilities["Pred"][:-1]  # Remove final comma
			predictions = prediction.split(",")

			''' Cull if parent node is not successful '''
			for i in range(0, len(children)):
				pred_children = list(set(predictions).intersection(set(children[i])))
				has_children = len(pred_children) > 0

				if p:
					if has_children and parents[i] not in predictions:
						predictions = list(set(predictions).difference(set(children[i])))
					elif has_children and parents[i] in predictions:
						predictions.remove(parents[i])

				''' Groups at the moment are mutually exclusive. Choose the subtype with the highest probability. '''
				if len(pred_children) > 1 and len(predictions) > 1:
					prob_no = probabilities[pred_children].apply(pd.to_numeric)
					max_subtype = [prob_no.idxmax()]
					drop_preds = list(set(prob_no.index.values).difference(set(max_subtype)))
					for drop_pred in drop_preds:
						try:
							predictions.remove(drop_pred)
						except ValueError:
							continue

			''' Post refinement, label cases if necessary '''
			if len(predictions) == 0:
				y_pred.append("Unclassified")
			elif len(predictions) > 1:  # Multi-label
				y_pred.append(','.join(predictions))
			else: # Single prediction
				y_pred.append(predictions[0])

		return y_pred

	def _filter_healthy(self, probabilities, counts):

		"""The idea for the healthy filter is simple.

		Add up genes we know are included in the immunophenotype,
		calculate the Z-score, and then for any new sample that passes through check whether it falls within 3
		standard deviations. Obviously this is fairly hamfisted, but it does the job for now. Given it's a user
		modified parameter, it can be run with and without and users can determine whether they can trust the result."""

		probabilities["B-ALL"] = "True"
		probabilities.loc[counts["B-ALL"] < -3, "B-ALL"] = "False"
		probabilities.loc[counts["B-ALL"] > 3, "B-ALL"] = "False"

		return probabilities


	def predict_proba(self, X, parents=False):

		"""
		Given a set of samples, return the probabilities of the classification attempt.

		...

		Parameters
		__________
		X : Pandas DataFrame
			Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).
		parents : bool
			True/False as to whether to include parents in the hierarchy in the output, i.e. Ph Group.

		Returns
		__________
		probabilities: Pandas DataFrame
			Probabilities returned by ALLSorts for each prediction - samples (rows) x subtype/meta-subtype (columns)
			Note: These do not have to add to 1 column-wise - see paper (when it is released!)
		"""

		counts = self._check_input(X)
		predicted_proba = pd.DataFrame(index=counts.index)
		coef_all = pd.DataFrame(columns=counts.columns)

		for subtype in self.fitted:

			if isinstance(self.genes, bool):
				genes = list(counts.columns)
			else:
				genes = self.genes[subtype]

			probs = self.fitted[subtype].predict_proba(counts[genes])
			probs = pd.DataFrame(probs,
								 columns=self.fitted[subtype].classes_,
								 index=counts.index)

			if "Others" in probs.columns:
				probs.drop(["Others"], axis=1, inplace=True)

			predicted_proba = pd.concat([predicted_proba, probs], axis=1, join="inner")


		predicted_proba = self._weighted_proba(predicted_proba)

		if not parents:
			predicted_proba.drop(self._get_parents(), inplace=True, axis=1)

		if self.filter_healthy:
			predicted_proba = self._filter_healthy(predicted_proba, counts)

		return predicted_proba

	def predict(self, X, probabilities=False, parents=True):

		"""
		Given a set of samples, return the predictions of the classification attempt.

		...

		Parameters
		__________
		X : Pandas DataFrame
			Pandas DataFrame that represents either:
			- The raw counts of your samples (rows) x genes (columns))
			- The probabilities as provided by predict_proba
		probabilities : bool
			True/False as to whether to indicate X is probabilities vs. raw counts
		parents : bool
			True/False as to whether to include parents in the hierarchy in the output, i.e. Ph Group.

		Returns
		__________
		predictions : Pandas DataFrame
			Predictions as made by ALLSorts given the input. A 1 x n Sample data Frame.
		"""

		if not probabilities:
			predicted_proba = self.predict_proba(X, parents=parents)
		else:
			predicted_proba = X.copy()

		predicted_proba["Pred"] = ""

		for subtype, pred_probs in predicted_proba.drop("Pred", axis=1).iteritems():
			predicted_proba.loc[pred_probs > self.thresholds[subtype], "Pred"] += subtype + ","

		y_pred = self._eval_preds(predicted_proba, p=parents)

		return y_pred

	def fit(self, X, y):

		""" Train the model relative to the input.

		Parameters
		__________
		X: Pandas DataFrame
			The training counts (samples/rows x genes/columns)
		y: Pandas Series
			The true labels for the training set.

		"""

		counts = self._check_input(X)

		self.f_hierarchy = self._get_flat_hierarchy()
		self._set_thresholds()

		# Train hierarchical classifier
		self._recurse_clf(self.hierarchy, counts, y)

		return self

