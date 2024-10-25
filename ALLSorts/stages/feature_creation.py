#=============================================================================
#
#   ALLSorts v2 - Feature Creation Stage
#   Author: Breon Schmidt
#   License: MIT
#
#=============================================================================

''' --------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import message, root_dir

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


import time

''' --------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------'''

class FeatureCreation(BaseEstimator, TransformerMixin):

	"""
	A class that represents a feature creation stage

	...

	Attributes
	__________
	kernel_div : int (5)
		# genes in a chromosome / kernel_div is a maximum size of a sliding
		window that the median iterative filter uses.
	n_jobs: int (1)
		Number of concurrent tasks to be run in parallel. Dependent on system.
	fusions: list (False)
		List of fusions with genes seperated with "_". The log difference will
		be taken in order to create a new feature, i.e. "GENE1_GENE2"

	Methods
	-------
	fit(counts, y)
		Get the median absolute devation of the training set.
	transform(counts)
		Create iAMP21_RATIO, chr1:22+X/Y, B_ALL features.
	fit_transform(counts, y=False)
		Apply fit and then transform.
	"""

	def __init__(self, kernel_div=5, n_jobs=1, fusions=False, fusion_feature=False,
				 iamp21_feature=True, chrom_feature=True):
		self.kernel_div = kernel_div
		self.n_jobs = n_jobs
		self.fusions = fusions
		self.fusion_feature = fusion_feature
		self.iamp21_feature = iamp21_feature
		self.chrom_feature = chrom_feature

	def _loadChrom(self):

		chrom_ref_path = str(root_dir())+"/data/chrom_refs.txt"

		self.chrom_ref = pd.read_csv(chrom_ref_path, sep="\t", header=None)
		self.chrom_ref.drop([1, 2, 5, 6, 7], axis=1, inplace=True)
		self.chrom_ref.columns = ["chrom", "start", "end", "meta"]
		self.chrom_ref["length"] = (self.chrom_ref["start"] -
									self.chrom_ref["end"]).abs()

		# Extract gene names
		gene_names = []
		for line, meta in self.chrom_ref["meta"].iteritems():
			gene_name = meta.split(";")[2].split('"')[1]
			gene_names.append(gene_name)

		self.chrom_ref["gene"] = gene_names
		self.chrom_ref.index = self.chrom_ref["chrom"]
		self.chrom_ref.drop(["chrom", "meta"], axis=1, inplace=True)
		self.chrom_ref["start"] = pd.to_numeric(self.chrom_ref["start"])
		self.chrom_ref["end"] = pd.to_numeric(self.chrom_ref["end"])

		# Create dictionary of genes per chromosome
		self.chrom_dict = {}
		for chrom, info in self.chrom_ref.iterrows():
			if chrom in self.chrom_dict:
				self.chrom_dict[chrom].append(info["gene"])
			else:
				self.chrom_dict[chrom] = [info["gene"]]

		self.chroms = list(range(1, 23)) + ["X", "Y"]

	def _median_filter(self, sample, size=5):
		filtered = median_filter(sample, mode="constant", size=size)
		return pd.Series(filtered)

	def _chromSmoothing(self, chrom, counts_norm):

		chrom_counts = counts_norm.reindex(self.chrom_dict[str(chrom)], axis=1).dropna(axis=1)

		q1 = chrom_counts.quantile(0.25, axis=1)
		q3 = chrom_counts.quantile(0.75, axis=1)
		iqr = q3 - q1
		upper = q3 + iqr * 1.5
		lower = q1 - iqr * 1.5

		c = np.matrix(chrom_counts)
		c_clip = c.transpose().clip(min=np.array(lower), max=np.array(upper))
		chrom_counts = pd.DataFrame(c_clip.transpose(), columns=chrom_counts.columns, index=chrom_counts.index)

		filt_columns = list(chrom_counts.columns)
		kernel_size = 5
		break_count = 0

		filtered = chrom_counts.copy()  # For first iteration
		while kernel_size <= len(filt_columns) / self.kernel_div:

			filtered = filtered.apply(self._median_filter, size=kernel_size, axis=1)
			filtered.columns = filt_columns

			kernel_size = int(kernel_size * 3)
			kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

			if break_count > 3:
				break
			else:
				break_count += 1

		return filtered

	def _mads(self, counts):

		mad_sub = self.mad.loc[counts.columns]
		mad_sub = mad_sub[mad_sub != 0]
		fcounts = counts[mad_sub.index]
		mads = np.subtract(np.matrix(fcounts), np.median(np.matrix(fcounts), axis=0))
		mads = pd.DataFrame(np.divide(mads, np.array(mad_sub)), index=counts.index, columns=mad_sub.index)

		return mads

	def _smoothSamples(self, counts):

		# Scale mads
		mads = self._mads(counts)

		# Smooth chromosomes
		smooth_chroms = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._chromSmoothing)
													 (chrom, mads)
													 for chrom in self.chroms)

		# Aggregate results
		smooth_samples = pd.DataFrame(index=counts.index)
		for smooth_chrom in smooth_chroms:
			smooth_samples = pd.concat([smooth_samples, smooth_chrom], axis=1, join="inner")

		return smooth_samples

	def _chromFeatures(self, scounts):

		chrom_features = pd.DataFrame(index=scounts.index)
		for chrom in self.chroms:
			chrom_ = str(chrom)
			genes = list(set(self.chrom_dict[chrom_]).intersection(
				set(scounts.columns)))

			if len(genes) != 0:
				chrom_features["chr" + chrom_] = list(scounts.loc[:, genes].median(axis=1))
			else:
				chrom_features["chr" + chrom_] = 0.0


		# For general
		chrom_high = []
		chrom_low = []

		for sample, chrom_meds in chrom_features.iterrows():
			chrom_high.append(chrom_meds.iloc[:-1][chrom_meds.iloc[:-1] > 0].sum())
			chrom_low.append(chrom_meds.iloc[:-1][chrom_meds.iloc[:-1] < 0].sum())

		chrom_features["med"] = chrom_features.median(axis=1)
		chrom_features["chr_high"] = chrom_high
		chrom_features["chr_low"] = chrom_low 
		chrom_features["chr_abs"] = chrom_features["chr_high"].abs() + chrom_features["chr_low"].abs()

		return chrom_features

	def _iamp21Feature(self, counts):

		bins = [15000000, 31514667, 43700713]
		bin_medians = pd.DataFrame(index=counts.index)

		chrom21 = self.chrom_ref[self.chrom_ref.index == "21"]
		mads = self._mads(counts)

		for i in range(0, len(bins)):

			# Get counts for genes in region
			if i == len(bins) - 1:
				bin_ = chrom21[(chrom21["start"] >= bins[i])]
			else:
				bin_ = chrom21[(chrom21["start"] >= bins[i]) &
							   (chrom21["start"] < bins[i + 1])]

			overlap = bin_["gene"].isin(list(mads.columns))
			bin_genes = list(bin_[overlap]["gene"])
			bin_counts = mads.loc[:, bin_genes]

			if bin_counts.shape[1] != 0: # There were no genes in this bin of chrom21 specified in input
				''' Smooth region '''
				bin_scounts = bin_counts.apply(self._median_filter, size=11, axis=1)
				bin_scounts.columns = bin_counts.columns

				''' Get region median and add to growing list of features'''
				bin_median = bin_scounts.median(axis=1)
				bin_median.name = "IAMP21_bin" + str(i + 1)
				bin_medians = pd.concat([bin_medians, bin_median],
										axis=1,
										join="inner")
			else:
				bin_medians["IAMP21_bin" + str(i + 1)] = 0.0

		iamp21_ratio = bin_medians.iloc[:, 1].sub(bin_medians.iloc[:, 2])
		bin_medians["IAMP21_ratio"] = iamp21_ratio

		return bin_medians

	def _immunoFeature(self, counts):

		all_genes = ["CD19", "CD34", "CD22", "DNTT", "CD79A"]
		try:
			all_immuno = pd.DataFrame(counts[all_genes].sum(axis=1), columns=["B-ALL"], index=counts.index)
		except KeyError:
			# column of 0's if any or all of immuno genes are missing
			# TODO print warning
			all_immuno =  pd.DataFrame(0, columns=["B-ALL"], index=counts.index)

		return all_immuno


	def _fusions(self, counts):

		fusions = pd.DataFrame(index=counts.index)

		for partners in self.fusions:
			gene_1 = partners.split("_")[0]
			gene_2 = partners.split("_")[1]
			try:
				fusions[partners] = counts[gene_1].sub(counts[gene_2])
			except:
				fusions[partners] = 0.0

		return fusions

	def _scale(self, counts):
		scaler = StandardScaler().fit(counts)
		return scaler.transform(counts)


	def fit(self, counts, y=False):

		self._loadChrom()
		self.mad = 1.4826 * counts.sub(counts.median()).abs().median()

		return self

	def transform(self, counts, y=False):

		scounts = self._smoothSamples(counts.fillna(0.0))
		counts_orig = counts.fillna(0.0)
		counts = pd.concat([counts,
							self._immunoFeature(counts_orig)],
							axis=1, join="inner")

		if self.fusion_feature:
			counts = pd.concat([counts,
								self._fusions(counts_orig)],
							   join="inner",
							   axis=1)

		if self.iamp21_feature:
			counts = pd.concat([counts,
								self._iamp21Feature(counts_orig)],
							   join="inner",
							   axis=1)

		if self.chrom_feature:
			counts = pd.concat([counts,
								self._chromFeatures(scounts)],
							   join="inner",
							   axis=1)

		return counts

	def fit_transform(self, counts, y=False):
		self.fit(counts, y)
		return self.transform(counts)
