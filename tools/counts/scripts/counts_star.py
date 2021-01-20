#=======================================================================================================================
#
#   ALLSorts v2 - The STAR aligner counts creator thingy! 
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
	cli.add_argument('-directory', '-d',
					 required=True,
					 help=("""Path to the STAR output with quantMode Gene parameter enabled.
							  I.e. *_ReadsPerGene.out.tab exist within this directory."""))

	cli.add_argument('-reference', '-r',
					 required=False,
					 help=("""List of genes and their aliases. Generally speaking, leave as the default."""))

	cli.add_argument('-strand', '-s',
				 required=True,
				 help=("""Please indicate whether the alignments are unstranded/reverse/forward strand. i.e. -strand no, (no, reverse, forward)"""))

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
	

def collate_counts(directory, strand):

	first = True
	progress = []

	if directory[-1] != "/":
		directory += "/"

	for file in glob.glob(directory+"*ReadsPerGene.out.tab"):

		name = file.split("/")[-1].split("_ReadsPerGene")[0]

		tab_open = pd.read_csv(file, sep="\t", index_col=0, header=None)
		tab_open.columns = ["no", "forward", "reverse"]
		tab_open.drop(["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"], inplace=True)
		
		if first:
			first = False
			counts = pd.DataFrame(index=tab_open.index)
			
		try:
			temp = pd.DataFrame(tab_open[user.strand])
		except:
			print("Strand information incorrect. Choose one of [no/forward/reverse]")
			sys.exit(0)

		temp.columns = [name]
		progress.append(temp)
		
	counts = pd.concat(progress, join="inner", axis=1)
	counts = counts.transpose()

	return counts

def format_counts(counts, gene_lookup):
	genes = harmoniser(counts.columns, gene_lookup)
	counts.columns = genes
	counts.drop("dropitlikeitshot", axis=1, inplace=True)
	counts = counts.groupby(axis=1, level=0).sum() # If multiple copies of genes, add together.
	counts = counts.astype(float)	

	return counts


''' --------------------------------------------------------------------------------------------------------------------
Run
---------------------------------------------------------------------------------------------------------------------'''

user = user_input()
counts = collate_counts(user.directory, user.strand)
gene_lookup = load_references(user.reference)
counts = format_counts(counts, gene_lookup)
counts.to_csv(user.output)




