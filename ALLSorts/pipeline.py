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
import pandas as pd
import plotly.express as px
from string import ascii_lowercase

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
	predict_plot, X, return_plot=False, comparison_dir=False)
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

		'''These should not be hard coded here, but... c'mon. I'm doing OK, give me a break.'''

		order = ["High Sig", "High hyperdiploid", 'Low hyperdiploid', "Near haploid", 'Low hypodiploid', 'Ph Group',
					"Ph-like", "Ph", "PAX5alt", 'ETV6-RUNX1 Group', 'ETV6-RUNX1', 'ETV6-RUNX1-like', 'KMT2A Group',
					'TCF3-PBX1', 'DUX4', 'iAMP21', 'NUTM1', 'BCL2/MYC', 'MEF2D', 'HLF', 'IKZF1 N159Y', 'PAX5 P80R',
					'ZNF384 Group']

		parents = {"Ph Group": ["Ph", "Ph-like"],
				   "ETV6-RUNX1 Group": ["ETV6-RUNX1", "ETV6-RUNX1-like"],
				   "High Sig": ["Low hyperdiploid", "High hyperdiploid", "Near haploid"]}

		'''Now, on with the show!'''

		thresholds = self.steps[-1][-1].thresholds
		probabilities = predicted_proba.copy()
		x = []
		y = []
		c = []
		sample_id = []
		true = []
		pred = []

		if "True" not in probabilities.columns:
			probabilities["True"] = ""

		for i, values in probabilities.drop(["Pred", "True"],  axis=1).iteritems():

			y += list(values)
			x += [i] * values.shape[0]
			sample_id += list(values.index)

			if i in parents.keys():
				labels = probabilities["True"].isin(parents[i])
			else:
				labels = probabilities["True"] == i

			true += list(probabilities["True"])
			pred += list(probabilities["Pred"])
			c += list(labels)

		title = "Probability Distributions (" + str(probabilities.shape[0]) + " Samples)"
		fig = px.strip(x=x, y=y, color=c,
					color_discrete_map={True: "#DD4075", False: "#164EB6"},
					hover_data={"sample": sample_id, "new_pred": pred},
					labels=dict(x="Subtype", y="Probability"), title=title).update_traces(
						jitter=1, marker={"size": 11, "line": {"color": "rgba(0,0,0,0.4)", "width":1}})

		fig.update_layout(font_size=42, title_font_size=72, yaxis={"dtick": 0.1},
						  plot_bgcolor='rgba(0,0,0,0)')

		fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dadada')

		for i in range(0, len(order)):
			fig.add_shape({'type': "line", 'x0': i - 0.5, 'x1': i + 0.5,
						   'y0': thresholds[order[i]], 'y1': thresholds[order[i]],
						   'line': {'color': '#03CEA4', 'width': 3}})

		if return_plot:
			return fig
		else:
			fig.show()


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

		if return_plot:
			return waterfall_plot
		else:
			waterfall_plot.show()

	def _plot_waterfall(self, prediction_order):

		meta_subtypes = ["High Sig", "ETV6-RUNX1 Group", "Ph Group"]
		waterfalls = prediction_order[~prediction_order["Pred"].isin(meta_subtypes)]
		no_samples = waterfalls[waterfalls["True"] == "Other"].shape[0]
		title = "Waterfall Distributions (" + str(no_samples) + " Samples)"
		order = list(waterfalls["Pred"].unique())
		if "Unclassified" in order:
			order.remove("Unclassified")
		thresholds = self.steps[-1][-1].thresholds

		'''Main Plot'''
		plt = px.bar(x=waterfalls.index, y=waterfalls["PPred"], color=waterfalls["True"],
					color_discrete_map=c_subtypes,
					hover_data={"Prediction": waterfalls["Pred"], "Ground Truth": waterfalls["True"]},
					labels=dict(x="Subtype", y="Probability"), title=title)
		plt.update_xaxes(categoryarray=waterfalls.index)
		plt.update_traces(marker={"line": {"width": 0}})

		'''Thresholds'''
		prev = -0.5
		for i in range(0, len(order)):
			no_samples = waterfalls[waterfalls["Order"] == order[i]].shape[0]
			plt.add_shape({'type': "line", 'x0': prev, 'x1': no_samples + prev,
						   'y0': thresholds[order[i]], 'y1': thresholds[order[i]],
						   'line': {'color': '#ffffff', 'width': 2}})

			plt.add_shape({'type': "line", 'x0': prev, 'x1': no_samples + prev,
						   'y0': -0.05, 'y1': -0.05,
						   'line': {'color': c_subtypes[order[i]], 'width': 15}})

			prev = prev + no_samples

		'''Colour the unclassifieds'''
		for sample, values in waterfalls.iterrows():
			if values["Pred"] == "Unclassified":
				pos = waterfalls.index.get_loc(sample)
				plt.add_shape({'type': "line", 'x0': pos - 1, 'x1': pos,
							   'y0': -0.05, 'y1': -0.05, "line": {"width": 0},
							   'line': {'color': "#ffffff", 'width': 35}})

			elif values["PPred"] < thresholds[values["Pred"]]:
				pos = waterfalls.index.get_loc(sample)
				plt.add_shape({'type': "line", 'x0': pos - 1, 'x1': pos,
							   'y0': -0.05, 'y1': -0.05, "line": {"width": 0},
							   'line': {'color': "#ffffff", 'width': 35}})

		'''Formatting'''
		plt.update_layout(font_size=24, title_font_size=40,
						  yaxis={"dtick": 0.1}, plot_bgcolor='#444444')
		plt.update_yaxes(showgrid=False)
		plt.update_xaxes(showticklabels=False)

		plt.add_annotation(x=0, y=-0.03, yref="paper", xref="paper",
						   text="Prediction", font_size=24,
						   showarrow=False)

		plt.add_shape({'type': "line", 'x0': -0.001, 'x1': 1.001, "xref": "paper",
					   "layer": "below", 'y0': -0.05, 'y1': -0.05, "line": {"width": 0},
					   'line': {'color': "#ffffff", 'width': 35}})

		return plt

	def _label_adjustment(self, predicted_proba):
		predicted_proba.to_csv("/home/hdang/test.tsv", sep="\t")
		adj_predicted_proba = predicted_proba.copy()

		'''Adjust ordering to account for unclassified and multi-class'''
		adj_predicted_proba["Order"] = adj_predicted_proba["Pred"]
		for sample, probs in adj_predicted_proba.iterrows():
			if "," in probs["Pred"]:
				calls = probs["Pred"].split(",")
				alpha = ascii_lowercase[:len(calls)]
				i = 0
				for call in calls:
					adj_predicted_proba.loc[sample + "_" + alpha[i]] = adj_predicted_proba.loc[sample].iloc[i]
					adj_predicted_proba.loc[sample+"_"+alpha[i], ["Pred", "Order"]] = call
					i += 1
				adj_predicted_proba.drop(sample, inplace=True)

			elif probs["Pred"] == "Unclassified":
				max_prob = probs.drop(["True", "Pred", "Order"]).sort_values(ascending=False).index[0]
				adj_predicted_proba.loc[sample, "Order"] = max_prob

		return adj_predicted_proba

	def _label_order(self, predicted_proba):


		prediction_order = pd.DataFrame(columns=predicted_proba.columns)

		for subtype in predicted_proba["Order"].value_counts().index:
			sub_probs = predicted_proba[predicted_proba["Order"] == subtype]
			sub_probs = sub_probs.sort_values(by=subtype, ascending=False)
			prediction_order = pd.concat([prediction_order, sub_probs], join="inner")

		prediction_order["PPred"] = ""
		prediction_order["POther"] = ""
		for sample, probs in prediction_order.iterrows():
			pred = prediction_order.loc[sample, "Order"]
			prediction_order.loc[sample, "PPred"] = prediction_order.loc[sample, pred]
			prediction_order.loc[sample, "POther"] = 1 - float(prediction_order.loc[sample, "PPred"])

		prediction_order = prediction_order[["Pred", "True", "PPred", "POther", "Order"]]

		prediction_order = prediction_order[~prediction_order["Order"].isin(
			["Ph Group", "ETV6-RUNX1 Group", "High Sig"]
		)]

		return prediction_order

	def predict_plot(self, X, return_plot=False, comparison_dir=False):

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
		comparison_dir : str
			Location of the comparison directory.

		Returns
		__________
		Matplotlib object containing the drawn figure

		Output
		__________
		UMAP Plot figure.

		"""
		# TODO: This is a hack to get the comparison directory. 
		# This should be the comparision results from retrained models
		if not comparison_dir:
			comparison_dir = os.path.join(str(root_dir()), "models", "allsorts", "comparisons")

		plt.figure(figsize=(20, 10))
		u = joblib.load(os.path.join(comparison_dir, "umap.sav"))
		c_labels = pd.read_csv(os.path.join(comparison_dir, "comparison_labels.csv"), index_col=0)
		c_labels = c_labels["labels"]
		c_genes = pd.read_csv(os.path.join(comparison_dir, "comparison_genes.csv"), index_col=0)
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



