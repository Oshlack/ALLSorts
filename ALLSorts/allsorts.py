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

# Operations
from ALLSorts.operations.train import train

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

# From command line
def run():
    """
        MODES OF OPERATION
        --
        There are two main ways ALLSorts can be used. It can be imported into a larger script or it can be executed from
        the command line. As running ALLSorts from the command line requires a set of parameters, we can say that if
        there are no parameters we can assume that we are importing (Of course, some may execute from the command line
        and forget to add any parameters.
    """
    ui = UserInput()

    if ui.cli:
        message(allsorts_asci)  # Greet user... because we're not savages.

        if ui.train:
            train_time = time.time()
            message("Training Mode", level=1)
            train(ui=ui)
            message("Total Train time " + str(round(time.time() - train_time, 2)))  # Seconds

        elif ui.comparison:

            message("Rebuilding Comparisons", level=1)

            message("Loading classifier...")
            allsorts_clf = joblib.load(str(root_dir())+"/models/allsorts/allsorts.pkl.gz")

            message("Making predictions...")
            probabilities = allsorts_clf.predict_proba(ui.samples, parents=False)
            predictions = pd.DataFrame(allsorts_clf.predict(ui.samples), columns=["Prediction"], index=ui.samples.index)

            probabilities["True"] = ui.labels
            probabilities["Pred"] = list(predictions["Prediction"])

            message("Building comparisons...")
            rebuild_comparisons(allsorts_clf, probabilities, ui)
            message("Finished.")


        else:
            message("Prediction Mode", level=1)
            message("Loading classifier...")
            allsorts_clf = joblib.load(str(root_dir())+"/models/allsorts/allsorts.pkl.gz")

            print(allsorts_clf)
            #allsorts_clf.save(path="allsorts.pkl.gz")

            # Create output
            output = ui.destination

            # Save out probabilities, predictions
            message("Saving predictions...")
            probabilities = allsorts_clf.predict_proba(ui.samples, parents=ui.parents)
            predictions = pd.DataFrame(allsorts_clf.predict(ui.samples), columns=["Prediction"], index=ui.samples.index)
            probabilities["Pred"] = predictions

            if not isinstance(ui.labels, bool):
                probabilities["True"] = list(ui.labels)

            probabilities.to_csv(ui.destination+"/probabilities.csv")
            predictions.to_csv(ui.destination+"/predictions.csv")

            # Save figures
            message("Saving figures...")

            '''
            umap_plot = allsorts_clf.predict_plot(ui.samples, return_plot=True)
            umap_plot.savefig(ui.destination + "/plot.png")
            '''

            if isinstance(ui.labels, pd.Series):
                dist_plot = allsorts_clf.predict_dist(ui.samples, ui.labels, parents=True, return_plot=True)
            else:
                dist_plot = allsorts_clf.predict_dist(ui.samples, ui.labels, parents=True, return_plot=True)
            dist_plot.savefig(ui.destination + "/distributions.png")

            if not isinstance(ui.labels, bool):
                waterfall_plot = allsorts_clf.predict_waterfall(ui.samples, ui.labels, return_plot=True)
            else:
                comparisons = pd.read_csv(str(root_dir())+"/models/allsorts/comparisons.csv", index_col=0)
                waterfall_plot = allsorts_clf.predict_waterfall(ui.samples, compare=comparisons, return_plot=True)
            waterfall_plot.savefig(ui.destination + "/waterfalls.png")


            # Goodbye
            message("Finished.")

    else:
        message("Missing parameters. Exiting.")


def load():
    allsorts_clf = joblib.load(str(root_dir()) + "/models/allsorts/allsorts.pkl.gz")

load()

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
