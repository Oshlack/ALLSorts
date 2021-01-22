#=======================================================================================================================
#
#   ALLSorts v2 - The Feature Counts aligner counts creator thingy! 
#	Note: Only for hg19
#
#   Author: Breon Schmidt
#   License: MIT
#
#	Input: user --help for all parameters
#	Output: Counts formatted for ALLSorts
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

import pandas as pd
import numpy as np
import glob
import sys, argparse
from pathlib import Path

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent

def user_input():

	cli = argparse.ArgumentParser(description="ALLSorts Counts CLI")
	cli.add_argument('-counts', '-c',
					 required=True,
					 help=("""Path to the Feature Counts csv."""))

	cli.add_argument('-reference', '-r',
					 required=False,
					 help=("""List of genes and their aliases. Generally speaking, leave as the default."""))

	cli.add_argument('-output', '-o',
				 required=True,
				 help=("""Please indicate the output path (/path/to/output/counts.txt)."""))

	user_input = cli.parse_args()
	return user_input


def load_references(gtf):

	if not gtf:
		gtf = str(root())+"/resources/genes_filtered.gtf"

	gene_lookup = pd.read_csv(gtf, 
							header=None, 
							index_col=0, 
							sep="\t").to_dict()[1]

	return gene_lookup


def harmoniser(genes, gene_info):
	
	update_genes = []
	for gene in list(genes):
		try:
			update_genes.append(gene_info[gene])
		except:
			update_genes.append("dropitlikeitshot")
			
	return update_genes
	

def load_counts(counts):

	counts_fc = pd.read_csv(counts, skiprows=1, sep="\t", index_col=0)
	counts_fc.drop(["Chr", "Start", "End", "Strand", "Length"], axis=1, inplace=True)
	filt_names = []
	for name in counts_fc.columns:
	    filt_names.append(name.split("/")[-1].split("_Aligned")[0])
	counts_fc.columns = filt_names
	counts_fc = counts_fc.transpose()

	return counts_fc


def format_counts(counts, gene_lookup):
	genes = harmoniser(counts.columns, gene_lookup)
	counts.columns = genes
	counts.drop("dropitlikeitshot", axis=1, inplace=True)
	counts = counts.groupby(axis=1, level=0).sum() # If multiple copies of genes, add together.

	return counts


''' --------------------------------------------------------------------------------------------------------------------
Run
---------------------------------------------------------------------------------------------------------------------'''

user = user_input()
counts = load_counts(user.counts)
gene_lookup = load_references(user.reference)
counts = format_counts(counts, gene_lookup)
counts.to_csv(user.output)




