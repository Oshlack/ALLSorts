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
from ALLSorts.common import message, create_dir, _flat_hierarchy

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
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, precision_recall_curve, auc
import pandas as pd
import numpy as np

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances

# Scoring+
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score

''' --------------------------------------------------------------------------------------------------------------------
Global Variable
---------------------------------------------------------------------------------------------------------------------'''

''' Horrifying solution, but alas, all I can think of without building my own gridsearch. '''
fold_predictions = ""
fold_count = 0
grid_total = 1
current_grid = 1
current_cv = 1

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

	''' Create results path '''
	search_path = ui.model_dir + "gridsearch/"
	create_dir([ui.model_dir, search_path])

	''' CV results storage '''
	thresholds_cv = {}
	results_cv = {"accuracy": [], "precision": [], "recall": [], "f1": []}

	'''Prepare for CV'''
	subtypes = list(ui.labels.unique())
	global fold_predictions, grid_total
	fold_predictions = pd.DataFrame(columns=["cv", "grid", "hierarchy", "standardisation", "chrom_feature",
											 "fusion_feature", "iamp21_feature", "centroid"] +
											subtypes + ["acc", "f1"])
	grid_total = ui.gcv

	'''CV Loop'''
	for fold in range(1, ui.cv + 1):

		message("Fold: " + str(fold))

		'''Outer loop - Fold Results'''
		seed = np.random.randint(1, 1000)
		x_train, x_test, y_train, y_test = train_test_split(ui.samples, ui.labels,
															stratify=ui.labels,
															test_size=0.2,
															random_state=seed)

		''' Inner loop - hyperparameter tuning '''
		allsorts_clf_fold = _tune(ui, x_train, y_train, fold=fold)  # This is the best estimator
		probabilities = allsorts_clf_fold.predict_proba(x_test, parents=True)
		f_hierarchy = allsorts_clf_fold.steps[-1][-1].f_hierarchy

		''' Optimise Prediction Thresholds '''
		thresholds = fit_thresholds(probabilities, f_hierarchy, y_test)
		for subtype, fold_thresh in thresholds.items():
			if subtype in thresholds_cv.keys():
				thresholds_cv[subtype].append(fold_thresh)
			else:
				thresholds_cv[subtype] = [fold_thresh]

		''' Score fold '''
		allsorts_clf_fold.steps[-1][-1].thresholds = thresholds
		y_pred = allsorts_clf_fold.predict(x_test, parents=True)
		results_cv = _score_fold(results_cv, y_test, y_pred, subtypes)

		'''Increment CV'''
		global current_cv, current_grid
		current_cv += 1
		current_grid = 1

	''' Train final model using all samples '''
	allsorts_clf = _tune(ui, ui.samples, ui.labels)

	if ui.payg:
		fold_predictions.to_csv("fold_predictons.csv")

	''' Average thresholds '''
	thresholds = {}
	for subtype, sub_thresh in thresholds_cv.items():
		thresholds[subtype] = round(sum(sub_thresh) / len(sub_thresh), 4)

	allsorts_clf.steps[-1][-1].thresholds = thresholds

	''' Save results and model '''
	_save_all(results_cv, allsorts_clf, ui)


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


	''' Build up the parameter arguments. For each hierarchy, we need a new parameter dict. '''
	allsorts_params = []

	for hierarchy in ui.hierarchy:

		''' Start building in the novel features '''

		# Fusions
		fusion_list = ["BCR_ABL1", "ETV6_RUNX1", "TCF3_PBX1", "TCF3_HLF"]

		# Base Model and params
		lr = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500,
								multi_class="auto", class_weight="balanced")

		lr_params = [{"C": 0.3}]
		classifier = HierarchicalClassifier()

		# Build a new parameter for this hierarchy
		allsorts_params += [
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

		if ui.baseline:
			allsorts_params += [
				{
					'standardisation': ["passthrough"],
					'centroids': ["passthrough"],
					'feature_select': [FeatureSelection(hierarchy=hierarchy, method="all", test=False)],
					'feature_create__chrom_feature': [False],
					'feature_create__iamp21_feature': [False],
					'feature_create__fusion_feature': [False],
					'train_model__hierarchy': [hierarchy],
					'train_model__model': [lr],
					'train_model__params': lr_params
				}
			]

	''' What is the most efficient way to parallelise this '''
	training_x_models = len(list(ParameterGrid(allsorts_params)))

	if training_x_models*ui.gcv >= ui.n_jobs and not ui.payg:
		grid_jobs = ui.n_jobs
		stage_jobs = 1
		scoring = "balanced_accuracy"
	else:
		grid_jobs = 1
		stage_jobs = ui.n_jobs
		scoring = b_accuracy

	''' Create Pipeline '''
	allsorts_pipe = ALLSorts([("preprocess", Preprocessing(filter_genes=True, norm="TMM")),
							  ("feature_create", FeatureCreation(n_jobs=stage_jobs, kernel_div=30, fusions=fusion_list)),
							  ("standardisation", Scaler()),
							  ("feature_select", FeatureSelection()),
							  ("centroids", CentroidCreate()),
							  ("train_model", classifier)], verbose=ui.verbose)

	''' Inform the user about the number of models being trained '''
	if fold == 1:
		message("Important: Training " + str(training_x_models) + " models (" +
				str(ui.gcv * ui.cv * training_x_models) + " with cross validation).", important=True)

	''' Perform Grid Search - Likely to take some time. '''
	allsorts_grid = GridSearchCV(allsorts_pipe, param_grid=allsorts_params,
								 cv=ui.gcv, n_jobs=grid_jobs,
								 scoring=scoring).fit(x_train, y_train)

	grid_results = _grid_save(allsorts_grid)
	grid_results.to_csv(ui.model_dir+"gridsearch/gridsearch_fold"+str(fold)+".csv")

	''' Pick the estimator that maximised the score in our gridsearchcv '''
	allsorts_clf = allsorts_grid.best_estimator_

	return allsorts_clf


def _save_all(results_cv, allsorts_clf, ui):

	""" Save the scores, model, and processed counts """

	'''Save Scores'''
	scores = pd.DataFrame(results_cv, index=list(range(1, ui.cv + 1)))
	scores.to_csv(ui.model_dir+"cross_val_results.csv")
	save_path_model = ui.model_dir + "allsorts.pkl.gz"

	'''Save Model'''
	message("Saving model to: " + save_path_model)
	allsorts_clf.save(path=save_path_model)

	''' Take the final model and save the processed counts as a csv. '''
	if ui.counts:
		p_counts = allsorts_clf.transform(ui.samples)
		p_counts["True"] = ui.labels
		p_counts["counts"].to_csv(ui.model_dir + "normed_counts.csv")


def _score_fold(results_cv, y_test, y_pred, subtypes):

	""" Calculate four summary statistics - Accuracy, Precision, Recall, and F1 """

	'''Calculate summary stats'''
	ac = round(accuracy_score(y_test, y_pred), 4)
	pr = round(precision_score(y_test, y_pred, average="weighted", zero_division=0, labels=subtypes), 4)
	re = round(recall_score(y_test, y_pred, average="weighted", zero_division=0, labels=subtypes), 4)
	f1 = round(f1_score(y_test, y_pred, average="weighted", zero_division=0, labels=subtypes), 4)

	'''Add to the growing list'''
	results_cv["accuracy"].append(ac)
	results_cv["precision"].append(pr)
	results_cv["recall"].append(re)
	results_cv["f1"].append(f1)

	return results_cv


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

	return pd.DataFrame(grid_results)


def b_accuracy(clf, X, y):

	global fold_predictions, fold_count, grid_total, current_grid, current_cv
	subtypes = list(fold_predictions.columns)
	new_fold = fold_count + 1
	new_id = str(current_cv) + "_" + str(new_fold)
	fold_count += 1

	fold_prediction = pd.DataFrame(0, columns=subtypes, index=[new_id])
	y_pred = list(clf.predict(X))
	y_true = y.copy()

	'''Create clf string'''
	clf_params = clf.get_params(deep=False)["steps"]
	fold_prediction.loc[new_id, "clf"] = str(clf_params)
	fold_prediction.loc[new_id, "grid"] = int(current_grid)
	fold_prediction.loc[new_id, "cv"] = int(current_cv)

	h = clf.steps[-1][-1].hierarchy
	if not h:
		h = "baseline"
	fold_prediction.loc[new_id, "hierarchy"] = str(h)

	for thing in clf_params:

		name = thing[0]
		param = thing[1]

		if name == "feature_create":
			creation = str(param).split("(")[1].split(",")
			for item in creation:
				if "feature" in item:
					hmm = item.replace(" ", "").replace("\n", "").replace("\t","")
					c_name = hmm.split("=")[0]
					c_value = hmm.split("=")[1]
					fold_prediction.loc[new_id, c_name] = c_value
				else:
					continue
		elif name == "standardisation":
			fold_prediction.loc[new_id, "standardisation"] = str(param)
		elif name == "centroids":
			if param != "passthrough":
				fold_prediction.loc[new_id, "centroid"] = "True"
			else:
				fold_prediction.loc[new_id, "centroid"] = "False"
		elif name == "centroids":
			if param != "passthrough":
				fold_prediction.loc[new_id, "centroid"] = "True"
			else:
				fold_prediction.loc[new_id, "centroid"] = "False"

	print(grid_total, current_grid, grid_total == current_grid)
	if grid_total == current_grid:
		current_grid = 1
	else:
		current_grid += 1

	for subtype in subtypes:

		if subtype in ["hierarchy", "acc", "f1", "cv", "grid", "standardisation", "chrom_feature", "fusion_feature",
					   "iamp21_feature", "centroid"]:
			continue

		labels = pd.DataFrame(list(y_true), index=y_true.index, columns=["True"])
		labels["Pred"] = y_pred
		labels = labels[labels["True"] == subtype]
		correct = labels[labels["True"] == labels["Pred"]].shape[0]

		fold_prediction.loc[new_id, subtype] = str(correct) + "/" + str(labels.shape[0])

	subtypes.remove("hierarchy")
	subtypes.remove("acc")
	subtypes.remove("f1")
	subtypes.remove("cv")
	subtypes.remove("grid")
	subtypes.remove("standardisation")
	subtypes.remove("chrom_feature")
	subtypes.remove("fusion_feature")
	subtypes.remove("iamp21_feature")
	subtypes.remove("centroid")

	fold_prediction.loc[new_id, "acc"] = balanced_accuracy_score(y, y_pred)
	fold_prediction.loc[new_id, "f1"]= f1_score(y, y_pred, average="weighted", zero_division=0, labels=subtypes)
	fold_predictions = pd.concat([fold_predictions, fold_prediction], join="inner")

	print(fold_predictions.iloc[-50:])

	return balanced_accuracy_score(y, y_pred)


