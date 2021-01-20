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
import time
import joblib
import pandas as pd

'''  Internal '''
from ALLSorts.common import message, root_dir
from ALLSorts.operations.comparisons import rebuild_comparisons
from ALLSorts.user import UserInput
from ALLSorts.operations.train import train

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

    if not ui:
        ui = UserInput()
        message(allsorts_asci)

    if ui.train:
        message("Training Mode", level=1)
        train_time = time.time()
        train(ui=ui)
        message("Total Train time " + str(round(time.time() - train_time, 2)))  # Seconds

    elif ui.comparison:
        message("Rebuilding Comparisons", level=1)

        allsorts_clf = load_classifier()
        allsorts_clf = _set_njobs(ui.n_jobs, allsorts_clf)
        allsorts_clf.steps[-1][-1].filter_healthy = True if ui.ball == "True" else False

        run_comparison_builder(ui, allsorts_clf)

    else:
        message("Prediction Mode", level=1)

        allsorts_clf = load_classifier()
        allsorts_clf = _set_njobs(ui.n_jobs, allsorts_clf)
        allsorts_clf.steps[-1][-1].filter_healthy = True if ui.ball == "True" else False

        run_predictions(ui, allsorts_clf)

def load_classifier(path=False):

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

    if not path:
        path = str(root_dir()) + "/models/allsorts/allsorts.pkl.gz"

    message("Loading classifier...")
    allsorts_clf = joblib.load(path)

    return allsorts_clf


def run_comparison_builder(ui, allsorts):

    """
    Build comparison results to compare to future predictions.

    I.e. what the waterfall plot displays in addition to the predicted sample probabilities.
    ...

    Parameters
    __________
    ui : User Input Class
        Carries all information required to execute ALLSorts, see UserInput class for further information.

    """

    predictions, probabilities = get_predictions(ui.samples, allsorts, labels=ui.labels, parents=True)
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


def run_predictions(ui, allsorts):

    """
    This is what we are here for. Use ALLSorts to make predictions!

    ...

    Parameters
    __________
    ui : User Input Class
        Carries all information required to execute ALLSorts, see UserInput class for further information.

    Output
    __________
    Probabilities.csv, Predictions.csv, Distributions.png, Waterfalls.png at the ui.destination path.

    """

    predictions, probabilities = get_predictions(ui.samples, allsorts, parents=ui.parents)
    probabilities["Pred"] = list(predictions["Prediction"])
    if not isinstance(ui.labels, bool):
        probabilities["True"] = ui.labels

    probabilities.round(3).to_csv(ui.destination + "/probabilities.csv")
    predictions.to_csv(ui.destination + "/predictions.csv")

    if "B-ALL" in probabilities.columns:
        get_figures(ui.samples, allsorts, ui.destination, probabilities.drop("B-ALL", axis=1))
    else:
        get_figures(ui.samples, allsorts, ui.destination, probabilities)

    message("Finished. Thanks for using ALLSorts!")


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


def get_figures(samples, allsorts, destination, probabilities, plots=["distributions", "waterfalls"]):

    """
    Make figures of the results.

    ...

    Parameters
    __________
    samples : Pandas DataFrame
        Pandas DataFrame that represents the raw counts of your samples (rows) x genes (columns)).
    destination : str
        Location of where the results should be saved.
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
            dist_plot.savefig(destination + "/distributions.png")

        if plot == "waterfalls":
            if "True" in probabilities.columns:
                comparisons = False
            else:
                comparisons = pd.read_csv(str(root_dir()) + "/models/allsorts/comparisons.csv", index_col=0)

            waterfall_plot = allsorts.predict_waterfall(probabilities, compare=comparisons, return_plot=True)
            waterfall_plot.savefig(destination + "/waterfalls.png")

        if plot == "manifold":
            umap_plot = allsorts.predict_plot(samples, return_plot=True)
            umap_plot.savefig(destination + "/manifold.png")


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
