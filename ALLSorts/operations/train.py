# =======================================================================================================================
#
#   ALLSorts v2 - Train model
#   Author: Breon Schmidt
#   License: MIT
#
# =======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''


'''  Internal '''
from ALLSorts.common import message, root_dir, create_dir, _flat_hierarchy

# ALLSorts pipeline and stages
from ALLSorts.pipeline import ALLSorts
from ALLSorts.stages.preprocessing import Preprocessing
from ALLSorts.stages.feature_creation import FeatureCreation
from ALLSorts.stages.feature_selection import FeatureSelection
from ALLSorts.stages.standardisation import Scaler
from ALLSorts.stages.centroids import CentroidCreate
from ALLSorts.stages.hierarchical import HierarchicalClassifier
from ALLSorts.operations.thresholds import fit_thresholds

'''  External '''
import sys, os
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import List, Any

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

# Scoring+
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def train(ui=False):

	''' TRAINING A MODEL (OUTER LOOP)
		--
		This operation requires two steps:
		1. With a tuned estimator returned from inner loop, calibrate optimal thresholds.
		2. Score this method
		3. Train a final model, using the average of the final thresholds.
	'''

	message("Cross Validation (this will take awhile):", level=2)

	# Create results path
	search_path = ui.model_dir + "gridsearch/"
	create_dir([ui.model_dir, search_path])

	# CV results
	subtypes = list(ui.labels.unique())
	thresholds_cv = {}
	results_cv = {}
	results_cv["accuracy"] = []
	results_cv["precision"] = []
	results_cv["recall"] = []
	results_cv["f1"] = []


	# First we need to fight out for the model of choice



	# Now we need to figure out thresholds

	for fold in range(1, ui.cv + 1):

		message("Fold: " + str(fold))
		seed = np.random.randint(1, 1000)
		x_train, x_test, y_train, y_test = train_test_split(ui.samples, ui.labels,
															stratify=ui.labels,
															test_size=0.2,
															random_state=seed)

		# Inner loop (hyperparameter tuning)
		allsorts_clf_fold = _tune(ui, x_train, y_train, fold=fold)

		probabilities = allsorts_clf_fold.predict_proba(x_test, parents=True)
		f_hierarchy = allsorts_clf_fold.steps[-1][-1].f_hierarchy

		# Optimise Prediction Thresholds
		thresholds = fit_thresholds(probabilities, f_hierarchy, y_test)
		allsorts_clf_fold.steps[-1][-1].thresholds = thresholds

		for subtype, fold_thresh in thresholds.items():
			if subtype in thresholds_cv.keys():
				thresholds_cv[subtype].append(fold_thresh)
			else:
				thresholds_cv[subtype] = [fold_thresh]

		# Score fold
		y_pred = allsorts_clf_fold.predict(x_test, parents=True)

		hierarchy = {
			"High Sig": {
				"High hyperdiploid": False,
				'Low hyperdiploid': False,
				"Near haploid": False
			},
			'Low hypodiploid': False,
			'iAMP21': False,
			'NUTM1': False,
			'BCL2/MYC': False,
			'TCF3-PBX1': False,
			'MEF2D': False,
			'HLF': False,
			'IKZF1 N159Y': False,
			'PAX5 P80R': False,
			'Ph Group': {
				"Ph-like": False,
				"Ph": False
			},
			"PAX5alt": False,
			'ETV6-RUNX1 Group': {'ETV6-RUNX1': False,
								 'ETV6-RUNX1-like': False},
			'ZNF384 Group': False,
			'KMT2A Group': False,
			'DUX4': False
		}

		f_hierarchy = _flat_hierarchy(hierarchy, flat_hierarchy={})
		probs = allsorts_clf_fold.predict_proba(x_test, parents=True)
		print(probs)
		print(f_hierarchy)
		fold_preds(y_test, probs, f_hierarchy=f_hierarchy)

		results_cv["accuracy"].append(round(accuracy_score(y_test, y_pred), 4))
		results_cv["precision"].append(round(precision_score(y_test, y_pred,
															 average="weighted", zero_division=0, labels=subtypes), 4))
		results_cv["recall"].append(round(recall_score(y_test, y_pred,
													   average="weighted", zero_division=0, labels=subtypes), 4))
		results_cv["f1"].append(round(f1_score(y_test, y_pred,
											   average="weighted", zero_division=0, labels=subtypes), 4))


	# Train final model using all samples
	allsorts_clf = _tune(ui, ui.samples, ui.labels)

	test = allsorts_clf.transform(ui.samples)
	test["True"] = ui.labels
	test["counts"].to_csv("normed_counts.csv")


	# Average thresholds
	thresholds = {}
	for subtype, sub_thresh in thresholds_cv.items():
		thresholds[subtype] = round(sum(sub_thresh) / len(sub_thresh), 4)

	allsorts_clf.steps[-1][-1].thresholds = thresholds

	# Save results and model
	scores = pd.DataFrame(results_cv, index=list(range(1, ui.cv + 1)))
	scores.to_csv(ui.model_dir+"cross_val_results.csv")

	save_path_model = ui.model_dir + "allsorts.pkl.gz"
	message("Saving model to: " + save_path_model)
	allsorts_clf.save(path=save_path_model)


def _tune(ui, x_train, y_train, fold="all"):

	''' TUNING A MODEL (INNER LOOP)
		--
		This operation requires two steps:
		1. Construct a pipeline using the ALLSorts class (A Sklearn pipeline extension).
		2. Gridsearch the parameter space that is outlined.

		Currently this is achieved by editing this function. Although, in future, this will be included within a
		a passable JSON file that contains the below. Given that this is likely only to be run once in a blue moon,
		this is not a priority.

		For those wishing to use the ALLSorts model, with some substitutions of algorithms, simply edit this file after
		making a copy of the original (save it somewhere so you can always revert). Note, setting up an ALLSorts
		pipeline and grid search is identical to setting up a usual sklearn pipeline.

		For more information on how to achieve this visit:
		https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
		https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	'''

	hierarchy = {
		"High Sig": {
			"High hyperdiploid": False,
			'Low hyperdiploid': False,
			"Near haploid": False
		},
		'Low hypodiploid': False,
		'iAMP21': False,
		'NUTM1': False,
		'BCL2/MYC': False,
		'TCF3-PBX1': False,
		'MEF2D': False,
		'HLF': False,
		'IKZF1 N159Y': False,
		'PAX5 P80R': False,
		'Ph Group': {
			"Ph-like": False,
			"Ph": False
		},
		"PAX5alt": False,
		'ETV6-RUNX1 Group': {'ETV6-RUNX1': False,
							 'ETV6-RUNX1-like': False},
		'ZNF384 Group': False,
		'KMT2A Group': False,
		'DUX4': False
	}

	f_hierarchy = _flat_hierarchy(hierarchy, flat_hierarchy={})


	### ADD Some features we want
	fusion_list: List[Any] = []
	fusion_list += ["BCR_ABL1"]
	fusion_list += ["ETV6_RUNX1"]
	fusion_list += ["TCF3_PBX1", "TCF3_HLF"]

	# Set parameters to be used in GridSearchCV
	lr = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500,
							multi_class="auto", class_weight="balanced")
	lr_params = [{"C": 0.3}]
	standard_params = []
	classifier = HierarchicalClassifier()

	# Create parameter grid for input into GridSearchCV
	allsorts_params = [

		# Hierarchy
		{
			'standardisation': [Scaler(scaler="std")],
			'centroids': [CentroidCreate(hierarchy=hierarchy, distance_function=euclidean_distances)],
			'feature_select': [FeatureSelection(hierarchy=hierarchy, method="all", test=False)],
			'feature_create__chrom_feature': [True],
			'feature_create__iamp21_feature': [True],
			'feature_create__fusion_feature': [True],
			'train_model__hierarchy': [hierarchy],
			'train_model__model': [lr],
			'train_model__params': lr_params
		}

	]

	# Note: Once benchmarks per cpu is made, estimate time of compute and distribute it accordingly
	training_x_models = len(list(ParameterGrid(allsorts_params)))
	grid_search_cv = 2


	if training_x_models*grid_search_cv >= ui.n_jobs:
		grid_jobs = ui.n_jobs
		stage_jobs = 1
	else:
		grid_jobs = 1
		stage_jobs = ui.n_jobs

	# Create Pipeline
	allsorts_pipe = ALLSorts([("preprocess", Preprocessing(filter_genes=True, norm="TMM")),
							  ("feature_create", FeatureCreation(n_jobs=stage_jobs, kernel_div=30, fusions=fusion_list)),
							  ("standardisation", Scaler()),
							  ("feature_select", FeatureSelection()),
							  ("centroids", CentroidCreate()),
							  ("train_model", classifier)], verbose=ui.verbose)


	# Check with user whether they want to train this many models
	if fold == 1:
		message("Important: Training " + str(training_x_models) + " models (" +
				str(grid_search_cv * ui.cv * training_x_models) + " with cross validation).", important=True)

	# Perform Grid Search - Likely to take a lot of time.
	allsorts_grid = GridSearchCV(allsorts_pipe, param_grid=allsorts_params,
								 cv=grid_search_cv, n_jobs=grid_jobs,
								 scoring="balanced_accuracy").fit(x_train, y_train)

	grid_results = _grid_save(allsorts_grid)
	grid_results.to_csv(ui.model_dir+"gridsearch/gridsearch_fold"+str(fold)+".csv")

	# Pick the estimator that maximised the score in our gridsearchcv
	allsorts_clf = allsorts_grid.best_estimator_

	return allsorts_clf

def _grid_save(grid_search):

	means = grid_search.cv_results_['mean_test_score']
	stds = grid_search.cv_results_['std_test_score']
	grid_results = {}

	for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):

		# Create headers
		for name, param in params.items():
			name_ = str(name.split("__")[1] if "__" in name else name)
			if name_ == "model":
				param_ = type(param).__name__
			elif name == "train_model__params":
				name_ = "model_param"
				param_ = str(param)
			elif name_ == "centroids":
				param_ = type(param).__name__
			elif name_ == "hierarchy":
				name_ = str(name)
				param_ = "flat"
				for subtype in param.keys():
					if isinstance(param[subtype], dict):
						param_ = "hierarchy"
						break
			else:
				param_ = str(param)

			if name_ not in grid_results.keys():
				grid_results[name_] = [param_]
			else:
				grid_results[name_].append(param_)

		if "mean_score" not in grid_results.keys():
			grid_results["mean_score"] = [mean]
			grid_results["std_score"] = [std]
		else:
			grid_results["mean_score"].append(mean)
			grid_results["std_score"].append(std)

	print(grid_results)

	return pd.DataFrame(grid_results)


def fold_preds(y, probabilities, f_hierarchy=False):
	weighted_count = 0
	no_subtypes = 0

	for subtype in f_hierarchy.keys():

		select = [subtype] if not f_hierarchy[subtype] else f_hierarchy[subtype]

		labels = y.copy()
		labels[labels.isin(select)] = 1
		labels[labels != 1] = 0

		no_samples = labels[labels == 1].shape[0]

		precision, recall, thresh = precision_recall_curve(list(labels), probabilities[subtype])
		weighted_count += auc(recall, precision)
		no_subtypes += 1

		print(subtype, auc(recall, precision), no_samples)


	return weighted_count / no_subtypes
