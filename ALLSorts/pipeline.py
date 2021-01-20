#=======================================================================================================================
#
#   ALLSorts v2 - The ALLSorts pipeline
#   Author: Breon Schmidt
#   License: MIT
#
#   Note: Inherited from Sklearn Pipeline
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import _flat_hierarchy, message, root_dir

''' External '''
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import time

''' --------------------------------------------------------------------------------------------------------------------
Global Variables
---------------------------------------------------------------------------------------------------------------------'''

c_subtypes = {
	# Greens
	"High Sig": "#358600",
	"High hyperdiploid": "#16a085",
	"Low hyperdiploid": "#1abc9c",
	"Low hypodiploid": "#27ae60",
	"Near haploid": "#2ecc71",
	"iAMP21": "#006266",

	# Purple
	"ETV6-RUNX1 Group": "#3C174F",
	"ETV6-RUNX1": "#9b59b6",
	"ETV6-RUNX1-like": "#8e44ad",

	# Red
	"PAX5alt": "#c0392b",
	"PAX5 P80R": "#e74c3c",

	# Blue
	"Ph Group": "#45aaf2",
	"Ph": "#2d98da",
	"Ph-like": "#3867d6",

	# Orange
	"KMT2A Group": "#e67e22",
	"KMT2A": "#e67e22",
	"KMT2A-like": "#f39c12",

	# Yellows
	"ZNF384 Group": "#ffd32a",
	"ZNF384": "#ffd32a",
	"ZNF384-like": "#ffdd59",

	# Others
	"DUX4": "#1e272e",  # Grey
	"HLF": "#FDA7DF",  # light pink
	"TCF3-PBX1": "#40407a",  # dark purple
	"IKZF1 N159Y": "#2c2c54",  # darkest purple
	"BCL2/MYC": "#22a6b3",  # dark cyan
	"NUTM1": "#B33771",  # light mauve
	"MEF2D": "#6D214F",  # dark mauve
	"IL3-IGH": "#000000",  # black
	"Unclassified": "#dddddd",
	"Other": "#ffffff"

}

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class ALLSorts(Pipeline):

	"""
	Fundamentally, ALLSorts is just a pipeline, consisting of stages that need to be executed sequentially.

	This prepares the input for training or prediction. This ALLSorts class extends the Scikit Learn pipeline
	class and contains all the sequential stages needed to run ALLSorts. It is ALLSorts!

	...

	Attributes
	__________
	Pipeline : Scikit-Learn pipeline class
		Inherit from this class.

	Methods
	-------
	transform(X)
		Execute every stage of the pipeline, transforming the initial input with each step. This does not include the
		final classification.
	predict_proba(X, parents=True)
		Prior to classification, transform the raw input appropriately.
	predict_dist(predicted_proba, return_plot=False, plot_height=7, plot_width=20)
		Generate probability distributions for the results derived from predict_proba.
	predict_waterfall(predicted_proba, compare=False, return_plot=False):
		Generate waterfall plots data for the results derived from predict_proba and comparisons.
	plot_waterfall(prediction_order)
		Generate waterfall plots from data generated in predict_waterfall.
	predict_plot, X, return_plot=False)
		Use UMAP to plot samples unto a predefined manifold.
	clone
		Create an empty clone of this pipeline
	save(path="models/allsorts.pkl.gz")
		Save pipeline in pickle format to the supplied path.

	"""

	def __init__(self, steps, memory=None, verbose=False):

		"""
		Initialise the class

		Attributes
		__________
		steps : list
			A list of all steps to be used in the pipeline (generally objects)
		memory : str or object
			Used to cache the fitted transformers. Default "None".
		verbose : bool
			Include extra messaging during the training of ALLSorts.
		"""

		self.steps = steps
		self.memory = memory
		self.verbose = verbose
		self._validate_steps()

	def _get_flat_hierarchy(self, hierarchy):
		return _flat_hierarchy(hierarchy, flat_hierarchy={})

	def _get_parents(self, f_hierarchy):
		parents = []
		for subtype in f_hierarchy:
			if f_hierarchy[subtype]:
				parents.append(subtype)

		return parents

	def transform(self, X):

		"""
		Transform the input through the sequence of stages provided in self.steps.

		The final step (the classification stage) will NOT be executed here.
		This merely gets the raw input into the correct format.
		...

		Parameters
		__________
		X : Pandas DataFrame
			Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).

		Returns
		__________
		Xt : Pandas DataFrame
			Transformed counts.
		"""

		Xt = X
		for _, name, transform in self._iter(with_final=False):
			Xt = transform.transform(Xt)

		return Xt

	def predict_proba(self, X, parents=True):

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

		Xt = self.transform(X)
		return self.steps[-1][-1].predict_proba(Xt, parents=parents)

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
			return self.steps[-1][-1].predict(self.transform(X), probabilities=probabilities, parents=parents)
		else:
			return self.steps[-1][-1].predict(X, probabilities=probabilities, parents=parents)

	def predict_dist(self, predicted_proba, return_plot=False, plot_height=7, plot_width=20):

		"""
		Given a set of predicted probabilities, generate a figure displaying distributions of probabilities.

		See https://github.com/Oshlack/AllSorts/ for examples.

		...

		Parameters
		__________
		predicted_proba : Pandas DataFrame
			Calculated probabilities via predict_proba.
		return_plot : bool
			Rather than showing the plot through whatever IDE is being used, send it back to the function call.
			Likely so it can be saved.
		plot_height : 7
			Height in inches of the final image.
		plot_width : 20
			Width in inches of the final image.

		Returns
		__________
		Matplotlib object containing the drawn figure

		Output
		__________
		Probability distribution figure.

		"""

		hierarchy = self.steps[-1][-1].hierarchy
		f_hierarchy = self._get_flat_hierarchy(hierarchy)
		thresholds = self.steps[-1][-1].thresholds
		probabilities = predicted_proba.copy()

		drop = ["Pred", "True"] if "True" in probabilities.columns else ["Pred"]

		''' Create prob. distribution for each subtype '''
		subtypes = probabilities.drop(drop, axis=1).columns.to_list()
		fig, ax = plt.subplots(figsize=(plot_width, plot_height))

		if "True" not in probabilities.columns:
			probabilities["True"] = ""

		for i in range(0, len(subtypes)):

			pos = i + 1
			select = [subtypes[i]] if f_hierarchy[subtypes[i]] is False else f_hierarchy[subtypes[i]]

			''' Mark subtypes positive if required '''
			true_subtype = probabilities.loc[probabilities["True"].isin(select), subtypes[i]]
			other_subtype = probabilities.loc[~probabilities["True"].isin(select), subtypes[i]]

			''' Plot both positive and negative '''
			x_jitter = np.random.uniform(pos - 0.25, pos, other_subtype.shape[0])
			ax.scatter(x=x_jitter, y=other_subtype.values, c="#333333", alpha=0.8)
			x_jitter = np.random.uniform(pos, pos + 0.25, true_subtype.shape[0])
			ax.scatter(x=x_jitter, y=true_subtype.values, c="#c0392b", alpha=0.8)

			''' Add thresholds '''
			thresh_x = np.linspace(pos - 0.25, pos + 0.25, 100)
			ax.plot(thresh_x, [thresholds[subtypes[i]]] * len(thresh_x), c="#22a6b3")

		''' Finalise plot '''
		ax.set_xticks(range(1, len(subtypes) + 1))
		ax.set_xticklabels(subtypes, rotation='vertical')
		ax.set_ylabel("Probability", fontsize=16)
		ax.set_xlabel("Subtypes", fontsize=16)
		plt.tight_layout()

		if return_plot:
			return plt
		else:
			plt.show()

	def predict_waterfall(self, predicted_proba, compare=False, return_plot=False):

		"""
		Given a set of predicted probabilities, generate a figure displaying the decreasing probabilities per sample.

		This depiction is useful to compare probabilities more directly, in an ordered way, as to judge the efficacy
		of the classification attempt.

		See https://github.com/Oshlack/AllSorts/ for examples.

		...

		Parameters
		__________
		predicted_proba : Pandas DataFrame
			Calculated probabilities via predict_proba.
		return_plot : bool
			Rather than showing the plot through whatever IDE is being used, send it back to the function call.
			Likely so it can be saved.
		compare : bool or Pandas DataFrame
			Samples to compare newly predicted samples to

		Returns
		__________
		Matplotlib object containing the drawn figure

		Output
		__________
		Waterfalls figure.

		"""

		probabilities = predicted_proba.copy()

		''' Have true labels been assigned? Are comparison samples provided? '''
		if "True" not in probabilities.columns:
			probabilities["True"] = "Other"

		if isinstance(compare, pd.DataFrame):
			probabilities = pd.concat([probabilities, compare], join="inner")

		''' Relabel (account for multiclass) and Order the samples according to probabilities'''
		prediction_results = self._label_adjustment(probabilities)
		prediction_order = self._label_order(prediction_results)

		'''Finally, plot'''
		waterfall_plot = self._plot_waterfall(prediction_order)
		plt.tight_layout()

		if return_plot:
			return waterfall_plot
		else:
			plt.show()

	def _plot_waterfall(self, prediction_order):


		meta_subtypes = ["High Sig", "ETV6-RUNX1 Group", "Ph Group"]
		prediction_order = prediction_order[~prediction_order["Pred"].isin(meta_subtypes)] # Not required

		ax = prediction_order.drop(["Pred", "True"], axis=1).plot(
			kind='bar', color=["r", "#555555"], stacked=True,
			figsize=(16, 6), align='edge', width=1)

		# Setup colour bar underneath plot
		x_ticks = ax.get_xticks()
		y_ticks = ax.get_yticks()

		true_labels = prediction_order["True"]
		predicted_labels = prediction_order["Pred"]

		for i, x_tick in enumerate(x_ticks[:-1]):

			ax.text(x_tick + 0.4, y_ticks[0] - 0.05, " ", size=0,
					bbox={'fc': c_subtypes[predicted_labels[i]], 'pad': 5,
						  "edgecolor": c_subtypes[predicted_labels[i]]})

		# Setup legend
		patchList = [mpatches.Patch(color=c_subtypes[subtype], label=subtype)
					 for subtype in c_subtypes if subtype in list(predicted_labels)]

		plt.legend(handles=patchList, loc="upper center", prop={'size': 9}, ncol=5,
				   bbox_to_anchor=(0.5, -0.1), fancybox=True)
		plt.gca().axes.get_xaxis().set_visible(False)

		# Set true colours
		for i in range(0, len(list(predicted_labels))):
			ax.get_children()[i].set_color(c_subtypes[predicted_labels[i]])
			ax.get_children()[i].set_color(c_subtypes[true_labels[i]])

		ax.set_ylabel("Probability", fontsize=16)
		ax.set_xlabel("Subtypes", fontsize=16)

		return plt

	def _label_adjustment(self, predicted_proba):

		adj_predicted_proba = predicted_proba.copy()

		'''Adjust ordering to account for unclassified and multi-class'''
		adj_predicted_proba["Order"] = adj_predicted_proba["Pred"]
		for sample, probs in adj_predicted_proba.iterrows():
			if "," in probs["Pred"]:
				adj_predicted_proba.loc[sample, "Order"] = "Multi"
			elif probs["Pred"] == "Unclassified":
				max_prob = probs.drop(["True", "Pred", "Order"]).sort_values(ascending=False).index[0]
				adj_predicted_proba.loc[sample, "Order"] = max_prob

		return adj_predicted_proba

	def _label_order(self, predicted_proba):


		prediction_order = pd.DataFrame(columns=predicted_proba.columns)

		for subtype in predicted_proba["Order"].value_counts().index:

			sub_probs = predicted_proba[predicted_proba["Order"] == subtype]
			if subtype == "Multi":
				multi = sub_probs
				continue

			sub_probs = sub_probs.sort_values(by=subtype, ascending=False)
			prediction_order = pd.concat([prediction_order, sub_probs], join="inner")


		if 'multi' in locals():
			prediction_order = pd.concat([prediction_order, multi], join="inner")
			prediction_order = prediction_order[prediction_order["Order"] != "Multi"]

		prediction_order["PPred"] = ""
		prediction_order["POther"] = ""
		for sample, probs in prediction_order.iterrows():
			pred = prediction_order.loc[sample, "Order"]
			prediction_order.loc[sample, "PPred"] = prediction_order.loc[sample, pred]
			prediction_order.loc[sample, "POther"] = 1 - float(prediction_order.loc[sample, "PPred"])

		prediction_order = prediction_order[["Pred", "True", "PPred", "POther"]]

		return prediction_order

	def predict_plot(self, X, return_plot=False):

		"""
		Given the raw counts, embed these within a UMAP visualisation consisting of the comparison data.

		...

		Parameters
		__________
		X : Pandas DataFrame
			Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).
		return_plot : bool
			Rather than showing the plot through whatever IDE is being used, send it back to the function call.
			Likely so it can be saved.

		Returns
		__________
		Matplotlib object containing the drawn figure

		Output
		__________
		UMAP Plot figure.

		"""

		plt.figure(figsize=(20, 10))
		u = joblib.load(str(root_dir()) + "/models/allsorts/comparisons/umap.sav")
		c_labels = pd.read_csv(str(root_dir()) + "/models/allsorts/comparisons/comparison_labels.csv", index_col=0)
		c_labels = c_labels["labels"]
		c_genes = pd.read_csv(str(root_dir()) + "/models/allsorts/comparisons/comparison_genes.csv", index_col=0)
		c_genes = list(c_genes.iloc[:, 0])

		u_c = u.embedding_
		X_t = self.transform(X)
		X_t = X_t["counts"].loc[:, c_genes]
		u_t = u.transform(X_t)

		plt.scatter(u_t[:, 0], u_t[:, 1], c="#000000", alpha=0.4)
		plt.scatter(u_c[:, 0], u_c[:, 1], c=[c_subtypes[r] for r in c_labels], alpha=0.4, marker="x")

		transformed_positions = pd.DataFrame(u_c)
		transformed_positions["label"] = list(c_labels)

		median = transformed_positions.groupby("label").median()
		for name, label in median.iterrows():
			plt.text(label[0], label[1], name, FontSize=16)

		if return_plot:
			return plt
		else:
			plt.show()

	def clone(self):

		""" Create an empty copy of this pipeline.

		Returns
		_________
		A clone of this pipeline
		"""

		return clone(self)

	def save(self, path="models/allsorts.pkl.gz"):

		""" Create an empty copy of this pipeline.

		Parameters
		__________
		path : str
			System path to save the picked object.

		Output
		_________
		Pickle the ALLSorts pipeline at the supplied path
		"""

		with open(path, 'wb') as output:
			joblib.dump(self, output, compress="gzip", protocol=-1)



