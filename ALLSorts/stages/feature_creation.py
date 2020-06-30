#=======================================================================================================================
#
#   ALLSorts v2 - Feature Creation Stage
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
from ALLSorts.common import message, root_dir

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class FeatureCreation(BaseEstimator, TransformerMixin):

    ''' Input: Pre-processed counts
        Output: Pre-processed counts with additional features.

        Methods: Filter > Normalise
    '''

    def __init__(self,
                 iter_smooth=5, kernel_div=5,
                 n_jobs=1, fusions=False):

        self.iter_smooth = iter_smooth
        self.kernel_div = kernel_div
        self.n_jobs = n_jobs
        self.fusions = fusions

    def _loadChrom(self):

        '''Load chromosome references

           Input: NULL
           Output: chrom reference and dictionary objects'''

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


    def _chromSmoothing(self, chrom, counts_norm):

        '''Apply median iterative smoothing to chromosomes'''

        chrom_counts = counts_norm.reindex(self.chrom_dict[str(chrom)], axis=1).dropna(axis=1)

        q1 = chrom_counts.quantile(0.25, axis=1)
        q3 = chrom_counts.quantile(0.75, axis=1)
        iqr = q3 - q1
        upper = q3 + iqr * 1.5
        lower = q1 - iqr * 1.5

        chrom_counts = chrom_counts.clip(upper=upper, axis=0)
        chrom_counts = chrom_counts.clip(lower=lower, axis=0)
        filt_columns = list(chrom_counts.columns)

        kernel_size = 5
        break_count = 0

        filtered = chrom_counts.copy()  # For first iteration
        while kernel_size <= len(filt_columns) / self.kernel_div:
            temp = filtered.apply(median_filter, mode="constant", size=kernel_size, axis=1)
            filtered = pd.DataFrame(temp, index=temp.index)
            filtered = filtered[0].apply(pd.Series)
            filtered.columns = filt_columns

            kernel_size = int(kernel_size * 3)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

            if break_count > 10:
                break
            else:
                break_count += 1

        return filtered

    def _smoothSamples(self, counts):

        '''Add custom chromosome features'''

        # Scale mads


        mad_sub = self.mad.loc[counts.columns]
        mads = (counts - counts.median()).div(mad_sub).dropna(axis=1)

        # Smooth chromosomes
        smooth_chroms = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._chromSmoothing)
                                                     (chrom, mads)
                                                     for chrom in self.chroms)

        # Aggregate results
        smooth_samples = pd.DataFrame(index=counts.index)
        for smooth_chrom in smooth_chroms:
            smooth_samples = pd.concat([smooth_samples, smooth_chrom], axis=1, join="inner")
        smooth_samples = smooth_samples.sub(smooth_samples.median(axis=1), axis=0)

        return smooth_samples

    def _chromFeatures(self, scounts):

        ''' From the smoothed counts calculate the medians per chromosome.
            Input: Null
            Output: Medians per chromosome
        '''

        chrom_features = pd.DataFrame(index=scounts.index)
        for chrom in self.chroms:
            chrom_ = str(chrom)
            genes = list(set(self.chrom_dict[chrom_]).intersection(
                set(scounts.columns)))

            chrom_features["chr" + chrom_] = list(scounts.loc[:, genes].median(axis=1))

        # For general
        chrom_features["chr_sum"] = chrom_features.sum(axis=1)

        # For Near haploid
        extremes = chrom_features.drop("chr_sum", axis=1)
        q1 = extremes.quantile(0.25, axis=1)
        q3 = extremes.quantile(0.75, axis=1)
        iqr = q3 - q1
        upper = q3 + iqr * 1.5
        lower = q1 - iqr * 1.5
        extremes = extremes[(extremes.gt(upper, axis=0)) |
                            (extremes.lt(lower, axis=0))].replace(np.nan, 0)
        chrom_features["chr_ext"] = extremes.sum(axis=1)

        return chrom_features

    def _iamp21Feature(self, counts):

        ''' Create the IAMP21 ratio
            Input: Null
            Output: Custom feature
        '''

        bins = [15000000, 31514667, 38190280, 43700713]
        bin_medians = pd.DataFrame(index=counts.index)

        chrom21 = self.chrom_ref[self.chrom_ref.index == "21"]
        mad_sub = self.mad.loc[counts.columns]
        mads = (counts - counts.median()).div(mad_sub).dropna(axis=1)

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

            # Smooth region
            temp = bin_counts.apply(median_filter, mode="constant", size=11, axis=1)
            bin_scounts = pd.DataFrame(temp,
                                       index=bin_counts.index)
            bin_scounts = bin_scounts[0].apply(pd.Series)
            bin_scounts.columns = bin_counts.columns

            # Get region median
            bin_median = bin_scounts.median(axis=1)
            bin_median.name = "IAMP21_bin" + str(i + 1)
            bin_medians = pd.concat([bin_medians, bin_median],
                                    axis=1,
                                    join="inner")

        # Return ratio
        iamp21_ratio = bin_medians.iloc[:, 1].sub(bin_medians.iloc[:, 3])
        bin_medians["IAMP21_ratio"] = iamp21_ratio

        return bin_medians


    def _immunoFeature(self, counts):

        all_genes = ["CD19", "CD34", "CD22", "DNTT", "CD79A"]
        all_immuno = pd.DataFrame(counts[all_genes].sum(axis=1), columns=["B-ALL"], index=counts.index)

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

    def visChroms(self, sample, figsize=(15, 5), meta=False, center=False):

        ''' Visualise the CNV for the chromosomes in a sample.

            Input: Sample to visualise in processed data
            Output: Chromosome plot for the sample
        '''

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        chroms
        self.scounts.loc[sample].plot.line(figsize=figsize, ax=ax, c="#CCCCCC", ylim=(-2, 2))
        new_labels = []

        chrom_start = 0
        no_genes = len(list(self.scounts.loc[sample].index))
        for chrom in self.chroms:
            # Segment each chromosome
            chrom_ = str(chrom)
            chrom_genes = list(set(self.chrom_dict[chrom_]).intersection(set(self.scounts.columns)))
            chrom_size = len(chrom_genes)
            chrom_end = chrom_size + chrom_start - 1
            plt.axvline(x=chrom_end, c="black")

            # Display median
            med = self.counts.loc[sample, "chr" + chrom_]
            plt.axhline(y=med, xmin=chrom_start / no_genes,
                        xmax=chrom_end / no_genes, c="purple")

            new_labels.append((chrom_end + chrom_start) / 2)
            chrom_start += chrom_size

        # Center lines
        if center:
            mid = self.counts.loc[sample, "chr1":"chrY"].mean()
            center = (self.counts.loc[sample, "chr1":"chrY"] - mid).mean()

            plt.axhline(y=mid, c="green")
            plt.axhline(y=center, c="blue")

        # Refine plot
        ax.set_xticks(new_labels)
        ax.set_xticklabels(chroms)
        plt.xlabel("Chromosome", fontsize=16)
        plt.suptitle("Gene Expression per Chromosome", fontsize=20)
        if meta:
            plt.title(meta)
        plt.ylabel("Median Absolute Deviation", fontsize=16)

        # Finalise
        plt.show()

    def fit(self, counts, y=False):

        self._loadChrom()
        self.mad = 1.4826 * counts.sub(counts.median()).abs().median()

        return self

    def transform(self, counts, y=False):

        scounts = self._smoothSamples(counts)
        counts = pd.concat([counts,
                            self._chromFeatures(scounts),
                            self._iamp21Feature(counts),
                            self._immunoFeature(counts)],
                            axis=1, join="inner")

        if self.fusions:
            counts = pd.concat([counts,
                                self._fusions(counts)],
                               join="inner",
                               axis=1)

        # If there is a discrepency between no. genes, fill with blanks
        missing_genes = list(set(self.mad.index).difference(counts.columns))
        if len(missing_genes) > 0:
            message("Note: " + str(len(missing_genes)) + " genes not found in  supplied samples, filling with zeroes. "
                    "This may impact classification performance. \n Follow the counts guide on Github (http://) "
                    "to resolve.")

            for gene in missing_genes:
                counts[gene] = 0


        return counts

    def fit_transform(self, counts, y=False):
        self.fit(counts)
        return self.transform(counts)
