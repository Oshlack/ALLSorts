#=======================================================================================================================
#
#   ALLSorts v2 - Command Line Interface
#   Author: Breon Schmidt
#   License: MIT
#
#   Parse user arguments
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''
import os

''' Internal '''
from ALLSorts.common import message, root_dir, get_hierarchy

''' External '''
import sys, argparse
import pandas as pd
from typing import List, Optional

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class UserInput:

    def __init__(self):
        if self._is_cli():
            self.cli = True
            self.input = self._get_args()

            ''' Data '''
            self.samples = self.input.samples
            self.labels = self.input.labels if self.input.labels else False
            
            '''Model Directory''' # This is going to be used in both training and prediction mode
            self.model_dir = os.path.join(str(root_dir()), "models", "allsorts") if not self.input.model_dir else self.input.model_dir

            '''Prediction Parameters '''
            self.parents = False if not self.input.parents else True
            self.ball = self.input.ball
            self.destination = False if not self.input.destination else self.input.destination

            '''Training Parameters'''
            self.train = False if not self.input.train else True
            self.cv = 3 if not self.input.cv else int(self.input.cv)
            self.gcv = 3 if not self.input.gcv else int(self.input.gcv)
            self.baseline = False if not self.input.baseline else True
            self.counts = False if not self.input.counts else True
            self.hierarchy = self._get_hierarchy() if self.input.hierarchy else False
            self.n_jobs = 1 if not self.input.njobs else int(self.input.njobs)
            self.verbose = False if not self.input.verbose else True
            self.force = False if not self.input.force else True
            self.payg = False if not self.input.payg else True
            self.gene_panel = self._parse_gene_panel(self.input.gene_panel) if self.input.gene_panel else None

            '''Misc'''
            self.test = self.input.test
            self.comparison = False if not self.input.comparison else True
            self._input_checks()
            self._load_samples()

        else:
            message("No arguments supplied. Please use allsorts --help for further information about input.")
            sys.exit(0)

    def _is_cli(self):
        return len(sys.argv) > 1

    def _get_args(self):

        ''' Get arguments and options from CLI '''

        cli = argparse.ArgumentParser(description="ALLSorts CLI")
        cli.add_argument('-samples', '-s',
                         required=True,
                         help=("""Path to samples (rows) x genes (columns) 
                                  csv file representing a raw counts matrix.

                                  Note: hg19 only supported currently, use 
                                  other references at own risk."""))

        cli.add_argument('-labels', '-l',
                         required=False,
                         help=("""(Optional) 
                                  Path to samples true labels. CSV with
                                  samples (rows) x [sample id, label] (cols).

                                  This will enable re-labelling mode.

                                  Note: labels must reflect naming conventions
                                  used within this tool. View the ALLSorts 
                                  GitHub Wiki for further details."""))

        cli.add_argument('-destination', '-d',
                         required=False,
                         help=("""Path to where you want the final
                                  report to be saved."""))

        cli.add_argument('-test', '-t',
                         required=False,
                         action='store_true',
                         help=("""Test will run a simple logistic regression."""))

        cli.add_argument('-train',
                         required=False,
                         action='store_true',
                         help=("""Train a new model. -labels/-l and -samples/-s must be set."""))

        cli.add_argument('-baseline',
                         required=False,
                         action='store_true',
                         help=("""Include a bare bones baseline (bbb) into the training gridsearch. """))

        cli.add_argument('-hierarchy',
                         required=False,
                         help=("""List of paths of the hierarchy of the models you wish to train. 
                                  -train -t flag must be set."""))

        cli.add_argument('-model_dir',
                         required=False,
                         help=("""Directory for a new model in training mode, 
                               or directory for a pre-trained model in prediction mode."""))

        cli.add_argument('-njobs', '-j',
                         required=False,
                         help=("""(int, default=1) Will set n_jobs for all Sklearn estimators/transformers."""))

        cli.add_argument('-cv',
                         required=False,
                         help=("""(int, default=3) If training, how many folds in the cross validation?"""))

        cli.add_argument('-payg',
                         required=False,
                         action="store_true",
                         help=("""(bool, default=False) Print as you go. """))

        cli.add_argument('-gcv',
                         required=False,
                         help=("""(int, default=3) If training, how many folds in the grid search?"""))

        cli.add_argument('-counts',
                         required=False,
                         action="store_true",
                         help=("""(bool, default=False) Output preprocessed counts."""))

        cli.add_argument('-verbose', '-v',
                         required=False,
                         action="store_true",
                         help=("""(flag, default=False) Verbose. Print stage progress."""))

        cli.add_argument('-comparison',
                         required=False,
                         action="store_true",
                         help=("""Rebuild comparisons for labelled visualisations."""))

        cli.add_argument('-force', '-f',
                         required=False,
                         action="store_true",
                         help=("""(flag, default=False) Force. Bypass warnings without user confirmation."""))

        cli.add_argument('-parents', '-p',
                         required=False,
                         action="store_true",
                         help=("""Include parent meta-subtypes in predictions. Note: This may remove previously 
                                  unclassified samples."""))

        cli.add_argument('-ball', '-b',
                         required=False,
                         help=("""(bool, default=True) Will include B-ALL flag in results."""))

        cli.add_argument('-gene_panel', '-gp',
                         required=False,
                         help=("""(str, default=None) A comma-separated list of genes, 
                                  or path to a file containing gene panel."""))

        user_input = cli.parse_args()
        return user_input

    def _parse_gene_panel(self, gene_panel_path: str) -> List[str]:
        """Parse a gene panel from a file path.

        Parameters
        ----------
        gene_panel_path : str
            Path to the gene panel file. Can be:
            - A CSV file with one gene per line
            - A CSV file with a header and gene names in a column
            - A comma-separated string of gene names

        Returns
        -------
        List[str]
            List of gene names

        Raises
        ------
        ValueError
            If the gene panel file cannot be parsed or is empty
        """
        try:
            # First try to parse as comma-separated string
            if ',' in gene_panel_path:
                genes = [g.strip() for g in gene_panel_path.split(',')]
                if genes:
                    message(f"Parsed {len(genes)} genes from comma-separated string")
                    return genes

            # Try to read as CSV file
            try:
                # Try reading as single column
                genes = pd.read_csv(gene_panel_path, squeeze=True).tolist()
            except:
                # Try reading as multi-column and take first column
                df = pd.read_csv(gene_panel_path)
                genes = df.iloc[:, 0].tolist()

            # Remove any empty strings or NaN values
            genes = [str(g).strip() for g in genes if str(g).strip()]

            if not genes:
                raise ValueError("No genes found in panel file")

            message(f"Parsed {len(genes)} genes from file: {gene_panel_path}")
            return genes

        except Exception as e:
            message(f"Error parsing gene panel file: {str(e)}", level="e")
            raise ValueError(f"Could not parse gene panel from: {gene_panel_path}")

    def _input_checks(self):

        if self.train and not self.hierarchy:
            self.hierarchy = self._get_hierarchy([str(root_dir())+"/models/hierarchies/phenocopy.txt",
                                                  str(root_dir())+"/models/hierarchies/flat.txt"])

        if self.train and not (self.labels and self.samples):
            message("Error: if -train is set both -labels/-l, -params/-p, -samples/-s must be also. Exiting.")
            sys.exit()

        if self.train and not self.input.model_dir:
            message("Error: if -train is set a model directory (-model_dir /path/to/model/) is required. Exiting.")
            sys.exit()
        # if not self.train and not self.destination:
        #     message("Error: if -train is not set a destination (-d /path/to/output/) is required. Exiting.")
        #     sys.exit()


    def _load_samples(self):

        if self.samples:
            self.samples = pd.read_csv(self.samples, index_col=0, header=0)

        if self.labels:
            self.labels = pd.read_csv(self.labels, index_col=0, header=None, squeeze=True)
            self.labels.name = "labels"

    def _get_hierarchy(self, paths):
        return get_hierarchy(paths)
