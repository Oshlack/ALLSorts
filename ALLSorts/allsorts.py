# ======================================================================================================================
#
#         ___                                    ___          ___          ___                   ___
#        /  /\                                  /  /\        /  /\        /  /\         ___     /  /\
#       /  /::\                                /  /:/_      /  /::\      /  /::\       /  /\   /  /:/_
#      /  /:/\:\   ___     ___  ___     ___   /  /:/ /\    /  /:/\:\    /  /:/\:\     /  /:/  /  /:/ /\
#     /  /:/~/::\ /__/\   /  /\/__/\   /  /\ /  /:/ /::\  /  /:/  \:\  /  /:/~/:/    /  /:/  /  /:/ /::\
#    /__/:/ /:/\:\\  \:\ /  /:/\  \:\ /  /://__/:/ /:/\:\/__/:/ \__\:\/__/:/ /:/___ /  /::\ /__/:/ /:/\:\
#    \  \:\/:/__\/ \  \:\  /:/  \  \:\  /:/ \  \:\/:/~/:/\  \:\ /  /:/\  \:\/::::://__/:/\:\\  \:\/:/~/:/
#     \  \::/       \  \:\/:/    \  \:\/:/   \  \::/ /:/  \  \:\  /:/  \  \::/~~~~ \__\/  \:\\  \::/ /:/
#      \  \:\        \  \::/      \  \::/     \__\/ /:/    \  \:\/:/    \  \:\          \  \:\\__\/ /:/
#       \  \:\        \__\/        \__\/        /__/:/      \  \::/      \  \:\          \__\/  /__/:/
#        \__\/                                  \__\/        \__\/        \__\/                 \__\/
#
#   Author: Breon Schmidt
#   License: MIT
#
# ======================================================================================================================

""" --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------"""


''' External '''
import os
import sys
import time
import joblib
import pandas as pd
import plotly
from typing import Optional, Union

'''  Internal '''
from ALLSorts.common import message, root_dir
from ALLSorts.operations.comparisons import rebuild_comparisons
from ALLSorts.user import UserInput
from ALLSorts.operations.train import train
from ALLSorts.pipeline import ALLSorts as allsorts_object

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def run(ui=False):

    """
    A function that runs ALLSorts in one of three modes: Training, Comparison adding, Prediction.

    - The Training mode will replace the model in the installed directory.
    - The Prediction mode will output a set of predictions and visualisations as per an input set of samples.
    - The Comparison mode will build comparisons from a supplied set of samples and labels to which to compare all
      new predictions. I.e. when no labels are u

    ...

    Parameters
    __________
    ui : User Input Class
        Carries all information required to execute ALLSorts, see UserInput class for further information.
    """

    if plotly.__version__ != "4.14.3":
        message("Incorrect Plotly version, please install version 4.14.3.")
        sys.exit()

    if not ui:
        ui = UserInput()
        message(allsorts_asci)
    model_path = os.path.join(ui.model_dir, "allsorts.pkl.gz")
    if ui.train:
        message("Training Mode", level=1)
        train_time = time.time()
        trained_allsorts = train(
            samples=ui.samples,
            labels=ui.labels,
            cv=ui.cv,
            gcv=ui.gcv,
            hierarchy=ui.hierarchy,
            gene_panel=ui.gene_panel,
            model_dir=ui.model_dir,
            save_model=True,
            save_counts=False,
            save_grid_results=True,
            payg=ui.payg,
            baseline=ui.baseline,
            n_jobs=ui.n_jobs,
            verbose=ui.verbose,
        )
        message("Total Train time " + str(round(time.time() - train_time, 2)))  # Seconds

    elif ui.comparison:
        message("Rebuilding Comparisons", level=1)
        allsorts_clf = load_classifier(path=model_path)
        allsorts_clf = _set_njobs(ui.n_jobs, allsorts_clf)
        allsorts_clf.steps[-1][-1].filter_healthy = True if ui.ball == "True" else False

        run_comparison_builder(ui, allsorts_clf)

    else:
        message("Prediction Mode", level=1)

        allsorts_clf = load_classifier(path=model_path)
        allsorts_clf = _set_njobs(ui.n_jobs, allsorts_clf)
        allsorts_clf.steps[-1][-1].filter_healthy = True if ui.ball == "True" else False

        message("Using thresholds:", level=2)
        message(allsorts_clf.steps[-1][-1].thresholds)

        # Run predictions and ignore return value in CLI mode
        # When used as a package, this function returns a dictionary of results
        results = run_predictions(
            allsorts=allsorts_clf,
            samples=ui.samples,
            labels=ui.labels,
            parents=ui.parents,
            save_results=True,
            save_counts=ui.counts,
            save_figures=True,
            destination=ui.destination,
            model_dir=ui.model_dir,
        )


def load_classifier(
        path: str = os.path.join(str(root_dir()), "models", "allsorts", "allsorts.pkl.gz"),
) -> allsorts_object:

    """
    Load the ALLSorts classifier from a pickled file.

    ...

    Parameters
    __________
    path : str
        Path to a pickle object that holds the ALLSorts model.
        Default: "/models/allsorts/allsorts.pkl.gz"

    Returns
    __________
    allsorts_clf : ALLSorts object
        ALLSorts object, unpacked, ready to go.
    """
    message(f"Loading classifier from {path} ...")
    allsorts_clf = joblib.load(path)

    return allsorts_clf


def run_comparison_builder(
        ui: UserInput,
        allsorts: allsorts_object,
):

    """
    Build comparison results to compare to future predictions.

    I.e. what the waterfall plot displays in addition to the predicted sample probabilities.
    ...

    Parameters
    __________
    ui : User Input Class
        Carries all information required to execute ALLSorts, see UserInput class for further information.

    """
    predictions, probabilities = get_predictions(
        ui.samples, allsorts=allsorts, labels=ui.labels, parents=True
    )
    probabilities["Pred"] = list(predictions["Prediction"])

    message("Building comparisons...")
    rebuild_comparisons(allsorts, probabilities, ui)
    message("Finished.")


def _set_njobs(n_jobs, classifier):

    ''' During training the number of threads is stored in the object that is saved out.

        Temporarily, this function will reset it to however threads the user will want to use. Ideally, the saved
        object should default to 1. But, as the only use of ALLSorts currently is through the command line, it is
        sufficient for now.'''

    for step in classifier.steps:
        try:
            step[-1].n_jobs = n_jobs
        except AttributeError:
            continue

    return classifier


def run_predictions(
        allsorts: allsorts_object,
        samples: pd.DataFrame,
        labels: Union[pd.Series, bool],
        parents: bool,
        *,
        save_results: bool = False,
        save_counts: bool = False,
        save_figures: bool = False,
        destination: Optional[str] = None,
        model_dir: Optional[str] = None,
):

    """
    This is what we are here for. Use ALLSorts to make predictions!

    ...

    Parameters
    __________
    allsorts : ALLSorts object
        The ALLSorts object to use for predictions.
    samples : pd.DataFrame
        The samples to make predictions on.
    labels : pd.Series
        The labels to use for predictions.
    parents : bool
        Whether to include parents in the predictions.
    save_results : bool
        Whether to save the predictions and probabilities.
    save_counts : bool
        Whether to save the normalised counts.
    save_figures : bool
        Whether to save the figures.
    destination : str
        The destination to save the results to.
    model_dir : str
        The model directory.

    Output
    __________
    Returns a dictionary with the predictions and probabilities.

    If save_results is True:
    Probabilities.csv, Predictions.csv, Distributions.png, Waterfalls.png at the ui.destination path.
    If save_counts is True:
        Processed_counts.csv at the ui.destination path.
    If save_figures is True:
        Distributions.png, Waterfalls.png at the ui.destination path.
    """

    predictions, probabilities = get_predictions(samples, allsorts, parents=parents)
    probabilities["Pred"] = list(predictions["Prediction"])
    if not isinstance(labels, bool):
        probabilities["True"] = labels

    if save_results:
        if not os.path.exists(destination):
            print(f"Creating directory {destination}")
            os.makedirs(destination)
        probabilities.round(3).to_csv(os.path.join(destination, "probabilities.csv"))
        predictions.to_csv(os.path.join(destination, "predictions.csv"))

    if save_counts:
        message("Saving normalised counts.")
        processed_counts = allsorts.transform(samples)
        processed_counts["counts"].to_csv(os.path.join(destination, "processed_counts.csv"))

    if save_figures:
        if "B-ALL" in probabilities.columns:
            get_figures(
                allsorts=allsorts,
                samples=samples,
                destination=destination,
                model_dir=model_dir,
                probabilities=probabilities.drop("B-ALL", axis=1),
                plots=["distributions", "waterfalls"],
            )
        else:
            get_figures(
                allsorts=allsorts,
                samples=samples,
                destination=destination,
                model_dir=model_dir,
                probabilities=probabilities,
                plots=["distributions", "waterfalls"],
            )

    message("Finished. Thanks for using ALLSorts!")
    return {"predictions": predictions, "probabilities": probabilities}


def get_predictions(samples, allsorts, labels=False, parents=False):

    """
    Given a set of samples use ALLSorts to return a set of predictions and probabilities.

    ...

    Parameters
    __________
    samples : Pandas DataFrame
        Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).
    labels : Pandas Series
        Pandas series that has a label associate with each sample.
    parents : bool
        True/False as to whether to include parents in the hierarchy in the output, i.e. Ph Group.

    Returns
    __________
    predictions: Pandas DataFrame
        A predictions for each inputted sample.
    probabilities: Pandas DataFrame
        Probabilities returned by ALLSorts for each prediction - samples (rows) x subtype/meta-subtype (columns)
        Note: These do not have to add to 1 column-wise - see paper (when it is released!)
    """

    message("Making predictions...")
    probabilities = allsorts.predict_proba(samples, parents=parents)
    if "B-ALL" in probabilities.columns:
        predictions = pd.DataFrame(allsorts.predict(probabilities.drop("B-ALL", axis=1), probabilities=True,
                                   parents=parents), columns=["Prediction"], index=samples.index)
    else:
        predictions = pd.DataFrame(allsorts.predict(probabilities, probabilities=True, parents=parents),
                                   columns=["Prediction"], index=samples.index)

    if isinstance(labels, pd.Series):
        probabilities["True"] = labels

    return predictions, probabilities


def get_figures(
        allsorts: allsorts_object,
        samples,
        destination,
        model_dir,
        probabilities,
        comparison_dir=False,
        plots=["distributions", "waterfalls"],
):

    """
    Make figures of the results.

    ...

    Parameters
    __________
    samples : Pandas DataFrame
        Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).
    destination : str
        Location of where the results should be saved.
    model_dir : str
        Location of the models directory.
    probabilities : Pandas DataFrame
        The result of running the get_predictions(samples, labels=False, parents=False) function.
        See function for further usage.
    plots : List
        List of plots required. Default:  "distributions", "waterfalls", and "manifold".
        See https://github.com/Oshlack/AllSorts/ for examples.

    Output
    __________
    Distributions.png, Waterfalls.png, Manifold.png at the ui.destination path.

    """

    message("Saving figures...")
    for plot in plots:

        if plot == "distributions":
            dist_plot = allsorts.predict_dist(probabilities, return_plot=True)
            dist_plot.write_image(os.path.join(destination, "distributions.png"), width=4000, height=1500, engine="kaleido")
            dist_plot.write_html(os.path.join(destination, "distributions.html"))

        if plot == "waterfalls":
            if "True" in probabilities.columns:
                comparisons = False
            else:
                try:
                    comparisons = pd.read_csv(os.path.join(model_dir, "comparisons.csv"), index_col=0)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Comparisons file not found in {model_dir}")

            waterfall_plot = allsorts.predict_waterfall(probabilities, compare=comparisons, return_plot=True)
            waterfall_plot.write_image(os.path.join(destination, "waterfalls.png"), height=900, width=2500, engine="kaleido")
            waterfall_plot.write_html(os.path.join(destination, "waterfalls.html"))

        if plot == "manifold":
            umap_plot = allsorts.predict_plot(samples, return_plot=True, comparison_dir=comparison_dir)
            umap_plot.savefig(os.path.join(destination, "manifold.png"))


''' --------------------------------------------------------------------------------------------------------------------
Global Variables
---------------------------------------------------------------------------------------------------------------------'''

allsorts_asci = """                                                            
       ,---.  ,--.   ,--.    ,---.                  ,--.         
      /  O  \ |  |   |  |   '   .-'  ,---. ,--.--.,-'  '-. ,---. 
     |  .-.  ||  |   |  |   `.  `-. | .-. ||  .--''-.  .-'(  .-' 
     |  | |  ||  '--.|  '--..-'    |' '-' '|  |     |  |  .-'  `)
     `--' `--'`-----'`-----'`-----'  `---' `--'     `--'  `----' 
    """
