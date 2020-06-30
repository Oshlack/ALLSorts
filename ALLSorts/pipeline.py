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
from ALLSorts.common import _flatHierarchy, message, root_dir

''' External '''
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

''' --------------------------------------------------------------------------------------------------------------------
Global Variables
---------------------------------------------------------------------------------------------------------------------'''

c_subtypes = {

    # Greens
    "High hyperdiploid": "#16a085",
    "Low hyperdiploid": "#1abc9c",
    "Low hypodiploid": "#27ae60",
    "Near haploid": "#2ecc71",
    "iAMP21": "#006266",

    # Purple
    "ETV6-RUNX1": "#9b59b6",
    "ETV6-RUNX1-like": "#8e44ad",

    # Red
    "PAX5alt": "#c0392b",
    "PAX5 P80R": "#e74c3c",

    # Blue
    "CRLF2(non-Ph-like)": "#45aaf2",
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

    def __init__(self, steps, hierarchy=False, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        if hierarchy:
            self.hierarchy = hierarchy
            self.f_hierarchy = self._flatHierarchy()

        self._validate_steps()

    def _flatHierarchy(self):
        return _flatHierarchy(self.hierarchy)

    def _getParents(self):
        parents = []
        for subtype in self.f_hierarchy:
            if self.f_hierarchy[subtype]:
                parents.append(subtype)

        return parents

    def transform(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        return Xt

    def predict_proba(self, X, parents=True):

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        return self.steps[-1][-1].predict_proba(Xt, parents=parents)

    def predict_dist(self, X, y=False, parents=False, return_plot=False):

        # Prepare figure
        fig, ax = plt.subplots(figsize=(20, 7))
        thresholds = self.steps[-1][-1].thresholds

        # Prepare Results
        predicted_proba = self.predict_proba(X, parents=True)
        predicted_proba["True"] = "" if isinstance(y, bool) else list(y)

        if not parents:
            predicted_proba.drop(self._getParents(), axis=1, inplace=True)

        # Create prob. distribution for each subtype
        subtypes = predicted_proba.drop("True", axis=1).columns.to_list()
        for i in range(0, len(subtypes)):
            pos = i + 1
            select = [subtypes[i]] if self.f_hierarchy[subtypes[i]] is False else self.f_hierarchy[subtypes[i]]

            # Mark subtypes positive if required
            true_subtype = predicted_proba.loc[predicted_proba["True"].isin(select), subtypes[i]]
            other_subtype = predicted_proba.loc[~predicted_proba["True"].isin(select), subtypes[i]]

            # Plot both positive and negative
            x_jitter = np.random.uniform(pos - 0.25, pos, other_subtype.shape[0])
            ax.scatter(x=x_jitter, y=other_subtype.values, c="#333333", alpha=0.8)
            x_jitter = np.random.uniform(pos, pos + 0.25, true_subtype.shape[0])
            ax.scatter(x=x_jitter, y=true_subtype.values, c="#c0392b", alpha=0.8)

            # Add thresholds
            thresh_x = np.linspace(pos - 0.25, pos + 0.25, 100)
            ax.plot(thresh_x, [thresholds[subtypes[i]]] * len(thresh_x), c="#22a6b3")

        # Finalise plot
        ax.set_xticks(range(1, len(subtypes) + 1))
        ax.set_xticklabels(subtypes, rotation='vertical')
        ax.set_ylabel("Probability", fontsize=16)
        ax.set_xlabel("Subtypes", fontsize=16)
        plt.tight_layout()

        if return_plot:
            return plt
        else:
            plt.show()

    def plot_waterfall(self, prediction_order):

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

    def predict_waterfall(self, X, y=False, compare=False, return_plot=False):

        prediction_results = self.predict_proba(X, parents=False)
        prediction_results["Pred"] = self.predict(X, parents=False)
        prediction_results["True"] = y if y is not False else "Other"

        if isinstance(compare, pd.DataFrame):
            prediction_results = pd.concat([prediction_results, compare], join="inner")

        prediction_results["Order"] = prediction_results["Pred"]

        for sample, probs in prediction_results.iterrows():
            if "," in probs["Pred"]:
                prediction_results.loc[sample, "Order"] = "Multi"
            elif probs["Pred"] == "Unclassified":
                max_prob = probs.drop(["True", "Pred", "Order"]).sort_values(ascending=False).index[0]
                prediction_results.loc[sample, "Order"] = max_prob

        # Order Probs
        prediction_order = pd.DataFrame(columns=prediction_results.columns)
        for subtype in prediction_results["Order"].value_counts().index:
            sub_probs = prediction_results[prediction_results["Order"] == subtype]
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
        waterfall_plot = self.plot_waterfall(prediction_order)
        plt.tight_layout()

        if return_plot:
            return waterfall_plot
        else:
            plt.show()

    def predict_plot(self, X, return_plot=False):

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

        plt.scatter(u_c[:, 0], u_c[:, 1], c=[c_subtypes[r] for r in c_labels], alpha=0.4, marker="x")
        plt.scatter(u_t[:, 0], u_t[:, 1], c="#000000")

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

        '''Create an empty copy of this pipeline. '''
        return clone(self)

    def save(self, path="models/allsorts.pkl.gz"):

        with open(path, 'wb') as output:
            joblib.dump(self, output, compress="bzip", protocol=-1)



