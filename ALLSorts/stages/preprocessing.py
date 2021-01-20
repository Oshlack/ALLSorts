#=============================================================================
#
#   ALLSorts v2 - Pre-processing Stage
#   Author: Breon Schmidt
#   License: MIT
#
#=============================================================================

''' --------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import message

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

''' --------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------'''

class TMM(BaseEstimator, TransformerMixin):

	"""
	A class that represents a normalised training set of counts.

	Based on the edgeR normalisation walkthrough provided by Josh Starmer (StatQuest), the extraordinary edgeR software,
	and Dr. Belinda Phipson.

	...

	Attributes
	__________
	BaseEstimator : Scikit-Learn BaseEstimator class
		Inherit from this class.
	ClassifierMixin : Scikit-Learn ClassifierMixin class
		Inherit from this class.
	
	Methods
	-------
	tmm_reference(counts)
		Chooses the sample in the training set to use as the TMM reference.
	scaling_factor(sample, reference)
		Determine the scaling factor for a sample relative to the reference.
	logCPM(counts, scaling_factors)
		Apply a logCPM transformation of scaled counts.
	fit(counts)
		Choose reference and library median to normalise by.
	transform(counts)
		apply TMM normalisation to raw counts.
	"""

	def __init__(self):

		"""
		Initialise the class

		Attributes
		__________
		method : str
			The method used to normalise (currently, only TMM)
		trained : bool
			Whether this object has normalised the training set and stored 
		"""

		self.method = "TMM"
		self.trained = False

	def tmm_reference(self, counts):

		""" Chooses sample in the supplied counts to use as the TMM reference.

		Parameters
		__________
		counts : Pandas DataFrame
			Gene expression counts (genes/columns x samples/rows)

		"""

		tmm_scaled = counts.divide(counts.sum(axis=1), axis=0)
		quantiles_75 = tmm_scaled.quantile(0.75, axis=1)
		quantiles_75_avg = quantiles_75.mean()

		# Choose reference based on 75th quantile closest to average
		quant_difference = (quantiles_75 - quantiles_75_avg).abs()
		reference_sample = quant_difference.sort_values().index[0]
		
		return counts.loc[reference_sample]

	def scaling_factor(self, sample, reference):

		""" Chooses sample in the supplied counts to use as the TMM reference.

		Parameters
		__________
		sample: Pandas Series
			The sample which we are trying to find a scaling factor for.
		reference: Pandas Series
			The reference we are comparing to (found with tmm_reference())
		"""

		sample_frame = pd.DataFrame(sample, index=sample.index)
		ref_frame = pd.DataFrame(reference, index=reference.index)
		counts = pd.concat([sample_frame, ref_frame], join="inner", axis=1)
		counts["multiple"] = np.array(counts.iloc[:, 0])*np.array(counts.iloc[:, 1])
		counts = counts[counts["multiple"] > 0]
		counts.columns = ["sample", "ref", "multiple"]

		''' Normalise by library size'''
		scaled_sample = counts["sample"]/sample.sum()
		scaled_ref = counts["ref"]/reference.sum()

		''' Remove biased genes '''
		scaled_ratio = scaled_sample/scaled_ref
		log_ratio = np.log2(scaled_ratio)
		log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan).dropna().sort_values()

		''' Remove genes with high/low expression in both samples '''

		geometric_means = np.log2(counts["multiple"])/2.0
		geometric_means = geometric_means.replace([np.inf, -np.inf], np.nan).dropna().sort_values()

		''' Now filter top/bottom by 30% and 5% respectively '''
		drop_genes = []
		cutoff_ratio_low = int(log_ratio.shape[0]*0.3)
		cutoff_geo_low = int(geometric_means.shape[0]*0.05) 
		cutoff_ratio_high = int(log_ratio.shape[0]*0.7) 
		cutoff_geo_high = int(geometric_means.shape[0]*0.95)
		
		drop_genes += list(log_ratio.iloc[0:cutoff_ratio_low].index)
		drop_genes += list(log_ratio.iloc[cutoff_ratio_high:-1].index)
		drop_genes += list(geometric_means.iloc[0:cutoff_geo_low].index)
		drop_genes += list(geometric_means.iloc[cutoff_geo_high:-1].index)

		''' Calculate weighted average of remaining log2 ratios '''
		scaling_factor = 2**log_ratio.drop(drop_genes).mean()
		
		return scaling_factor

	def logCPM(self, counts, scaling_factors):

		""" Apply a logCPM transformation of scaled counts.

		Parameters
		__________
		sample: Pandas Series
			The sample which we are trying to find a scaling factor for.
		reference: Pandas Series
			The reference we are comparing to (found with tmm_reference())
		"""
		
		lib_sizes = counts.sum(axis=1)

		lib_size_scaled = lib_sizes*scaling_factors
		prior_count_scaled = lib_size_scaled/lib_size_scaled.mean() * 0.5
		
		if not self.trained:
			self.library_median = lib_size_scaled.median()
			self.trained = True
			
		lib_size_scaled = lib_size_scaled + 2*prior_count_scaled

		counts_sum = counts.add(prior_count_scaled, axis=0)
		tmm_cpm = np.log2(counts_sum.div(lib_size_scaled, axis=0)*self.library_median)

		return tmm_cpm

	def fit(self, counts):

		"""Choose reference and library median to normalise by.

		Parameters
		__________
		sample: Pandas Series
			The sample which we are trying to find a scaling factor for.
		reference: Pandas Series
			The reference we are comparing to (found with tmm_reference())
		"""

		self.tmm_ref = self.tmm_reference(counts)
		raw_scaling_factors = counts.apply(self.scaling_factor, axis=1, reference=self.tmm_ref)
		self.scaling_geometric = np.exp(np.log(raw_scaling_factors).mean())

		return self	

	def transform(self, counts):

		""" Apply TMM normalisation to raw counts. 

		Parameters
		__________
		counts: Pandas DataFrame
			Apply TMM normalisation to raw counts (samples/rows x genes/cols).
		"""

		raw_scaling_factors = counts.apply(self.scaling_factor, axis=1, reference=self.tmm_ref)
		scaled_scaling_factors = raw_scaling_factors/self.scaling_geometric
		tmm_cpm = self.logCPM(counts, scaled_scaling_factors)

		return tmm_cpm

class Preprocessing(BaseEstimator, TransformerMixin):

	"""
	A class that represents the preprocessing result on at set of counts.

	...

	Attributes
	__________
	method : str
		The method used to normalise (currently, only TMM)

	Methods
	-------
	filter_cpm(counts, y)
		filter genes using a CPM cutoff (each sample avg. 10 reads).
	fit(counts, y)
		Get all preprocessing parameters relative to the training set.
	transform(counts)
		Transform input counts by parameters determined by training set.
	fit_transform(counts, y=False)
		Apply fit and then transform.
	"""

	def __init__(self, filter_genes=True, norm="TMM"):
		self.filter_genes = filter_genes
		self.norm = norm

	def _filter(self, gene, cutoff, sample_no):
		
		if gene[gene > cutoff].shape[0] > sample_no:
			return gene.name

	def filter_cpm(self, counts, y):

		cpm = counts.div(counts.sum(axis=1), 0)*1000000
		cutoff = 10/(counts.sum(axis=1).min()/1000000)
		min_samples = y.value_counts().min()

		filtered_genes = cpm.apply(self._filter, axis=0, cutoff=cutoff, sample_no=min_samples)
		self.genes = list(filtered_genes.dropna().index)

	def fit(self, counts, y):

		""" Get all preprocessing parameters relative to the training set.

		Parameters
		__________
		counts: Pandas DataFrame
			The training counts (samples/rows x genes/columns)
		y: Pandas Series
			The true labels for the training set.
		"""

		if self.filter_genes:
			self.filter_cpm(counts, y)
			counts = counts[self.genes]

		if self.norm == "TMM":
			self.tmm_norm = TMM().fit(counts)

		return self

	def transform(self, counts, y=False):

		""" Pre-process input counts as per parameters determined by fit().

		Parameters
		__________
		counts: Pandas DataFrame
			The training counts (samples/rows x genes/columns)
		"""

		counts.index = counts.index.astype("str")

		''' Filter genes '''
		if self.filter_genes:
			counts = counts.reindex(self.genes, axis=1)

		''' Normalise with TMM '''
		if self.norm == "TMM":
			counts = self.tmm_norm.transform(counts)

		''' Check for missing genes '''

		missing_genes = list(set(self.genes).difference(counts.columns))
		if len(missing_genes) > 0:
			message("Note: " + str(len(missing_genes)) +
					" genes not found in supplied samples, filling with zeroes.\n" +
					"This WILL impact classification performance.\n" +
					"Follow the counts guide on Github (http://) to resolve.", level="w")

		return counts

	def fit_transform(self, counts, y=False):

		""" Apply fit and then transform.

		Parameters
		__________
		counts: Pandas DataFrame
			The training counts (samples/rows x genes/columns)
		y: Pandas Series
			The true labels for the training set.
		"""

		self.fit(counts, y)
		return self.transform(counts)
