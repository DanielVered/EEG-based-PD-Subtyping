import os
import mne
import scipy
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, BaseCrossValidator
from sklearn.base import clone
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    adjusted_rand_score
)

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway, chi2_contingency

SEED = 42


class LeaveOneOutTrainEval(BaseCrossValidator):
    """
    A Leave-One-Out (LOO) like CrossValidator object that fits and scores on each N-1 samples in the data.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(X)

    def split(self, X, y=None, groups=None):
        n = len(X)
        idx = np.arange(n)
        for holdout in range(n):
            train_idx = np.delete(idx, holdout)
            yield train_idx, train_idx


class IOUtilz:
    @staticmethod
    def read_clinical_features(path: str) -> pd.DataFrame:
        """
        Reads the data from the clinical features .xlsx into a dataframe.
        :param path: a path to the .xlsx file of the clinical data.
        """
        data_file = pd.ExcelFile(path)
        cl_data = pd.concat([data_file.parse(sheet_name=sheet) for sheet in data_file.sheet_names])
        cl_data['IsHC'] = cl_data['Subject'].str.startswith('HC')
        cl_data.loc[~cl_data['IsHC'], 'Group'] = cl_data.loc[~cl_data['IsHC'], 'Group'].fillna('Idiopathic')
        return cl_data.reset_index(drop=True)


class PreprocessUtilz:
    @staticmethod
    def normalize_col(col: pd.Series) -> float:
        """
        Performs z-score normalization of a given column (pandas series).
        """
        return (col - col.mean()) / col.std()

    def normalize_group_means(means: np.ndarray | pd.Series, ses: np.ndarray | pd.Series = None,
                              mode: str = None) -> tuple[np.ndarray | pd.Series]:
        """
        Normalize the means and standard errors (ses) of a given feature across groups (e.g. clusters).
        :param means: a sequence representing the mean of a given feature for each group.
        :param ses: a sequence representing the standard error of a given feature for each group.
        :param mode: the name of the normalization mode, should be one of 'minmax', 'z-score'.
        """
        if mode is None:
            return means, ses

        if mode.lower() == "minmax":
            fmin = means.min(axis=0)
            rng = (means.max(axis=0) - fmin)
            rng = rng.replace(0, 1.0)
            means_n = (means - fmin) / rng
            ses_n = None if ses is None else ses / rng
            return means_n, ses_n

        if mode.lower() == "z-score":
            mu = means.mean(axis=0)
            sigma = means.std(axis=0).replace(0, 1.0)
            means_n = (means - mu) / sigma
            ses_n = None if ses is None else ses / sigma
            return means_n, ses_n

        raise ValueError("mode must be None, 'minmax', or 'z-score'")

    @staticmethod
    def _preprocess_pipeline(pipe, data: np.ndarray):
        """
        Transforms the input data along all pipeline steps excluding the last one (e.g. clustering model).
        :param pipe: a Pipeline object representing the model.
        :param data: an input numpy array for the model.
        """
        return pipe[:-1].transform(data)

    @staticmethod
    def set_random_seeds(models: list, seed: int = None):
        """
        Sets a random seed to all steps in a the given pipelines.
        :param models: a list of pipeline objects.
        :param seed: a random seed to set. Default is None.
        """
        for m in models:
            params_to_set = {}

            for name, step in m.steps:
                if hasattr(step, 'random_state'):
                    param_key = f"{name}__random_state"
                    params_to_set[param_key] = seed

            if params_to_set:
                m.set_params(**params_to_set)

    @staticmethod
    def get_ylim(data: pd.DataFrame, feature_name: str, groupby: list[str], offset: bool = True
                 , error: str = 'se') -> (float, float):
        """
        Calculates scalar limits for a given feature_name in the data using groupby columns.
        :param data: a dataframe with the relevant columns.
        :param feature_name: the feature name to get scalar limits for.
        :param groupby: a list of column names according to which to group the data and calculate the limits.
        :param offset: if True, widen the scalar limits in both directions by a standard deviation offset.
        :param error:
        :return ylim: a tuple (low, high) of the scalar limits for the given feature.
        """
        groupby = data.groupby(groupby)[feature_name]
        means = groupby.mean()
        ylim = means.min(), means.max()

        if offset:
            if error == 'se':
                offset = groupby.sem().max()
            else:
                offset = groupby.std().max()
            ylim = ylim[0] - 2 * offset, ylim[1] + 2 * offset

        return ylim


class TopoPlot:
    def __init__(self, data: pd.DataFrame, group_col: str, channel_col: str, feature_names: list[str],
                 display_names: list[str] = None, montage: str = 'standard_1020', cmap: str = 'viridis',
                 dims: (float, float) = (3, 4), vlim: (float, float) = None, title: str = None):
        """
        :param data: a dataframe with the raw feature table.
        :param group_col: the name of a column in the feature table to split the figures by.
        :param channel_col: the name of a column in the feature table which corresponds to the channel name \ ID.
        :param feature_names: a list of feature name to analyze topologically.
        :param display_names: the display names for the given features (labels of color bars).
        :param montage: a name of an MNE-EEG montage for the topological figure.
        :param cmap: a seaborn colormap name.
        :param dims: a tuple of dimensions for each figure.
        :param title: a customized title for the figure.
        """
        self.features = data.copy()
        self.feature_names = feature_names

        if display_names is None:
            self.display_names = [f.replace('_', ' ').strip() for f in feature_names]
        else:
            self.display_names = display_names

        self.channel_col = channel_col
        self.group_col = group_col
        self.groups = self.features[self.group_col].unique()
        self.n_groups = len(self.groups)

        self.eeg_info = TopoPlot.get_eeg_info(
            electrodes=self.features[self.channel_col].unique()
            , montage=montage
        )

        self.cmap = cmap
        self.dims = dims

        self.vlim = vlim
        self.title = title

        self.plot_topomap_mat()

    @staticmethod
    def get_eeg_info(electrodes: np.ndarray | list, montage: str):
        """
        Creates an MNE info object for downstream MNE topological plots creation.
        :param electrodes: a sequence of electrode names for the given configuration.
        :param montage: the desired configuration of electrodes on the scalp.
        """
        ch_names = list(electrodes)
        info = mne.create_info(ch_names=ch_names, sfreq=1, ch_types='eeg')
        montage = mne.channels.make_standard_montage(montage)
        info.set_montage(montage, on_missing='ignore')
        return info

    def plot_topomap(self, features: pd.DataFrame, feature_name: str, ax, title: str = '', vlim: (float, float) = None):
        """
        Creates figure with the topological distribution of the given feature name over the scalp using '.montage'.
        :param features: a transformed feature table (dataframe) with features are columns.
        :param feature_name: the feature name to plot its topological distribution.
        :param channel_col:
        :param ax: a matplotlib axes to draw the plot within.
        :param title: a title for the given axes.
        :param vlim: a tuple of floats (min, max) representing the color bar limits.
        If not given, estimated from values using PreprocessUtilz.get_ylim.
        """
        if vlim is None:
            vlim = PreprocessUtilz.get_ylim(features
                                            , feature_name
                                            , self.channel_col
                                            , offset=False
                                            , error='se')

        values = features.groupby(self.channel_col)[feature_name].mean().values
        im, cn = mne.viz.plot_topomap(values
                                      , self.eeg_info
                                      , axes=ax
                                      , cmap=self.cmap
                                      , vlim=vlim
                                      , show=False
                                      )
        ax.set_title(title, fontsize=13, y=1.02)
        return im

    def plot_topomap_row(self, feature_name: str, display_name: str, group_names: bool = True, axes=None):
        """
        Creates figure with the topological distribution of the given feature name over the scalp using '.montage'.
        The figure consists of a 1D row of subplots, one per each group, defined using '.group_col'.
        :param feature_name: the feature name to plot its topological distribution.
        :param display_name: the display name for the given feature (label of the color bar).
        :param group_names: If true, adds groups names as titles to the subplots. Default is True.
        :param axes: a 1D matplotlib axes for plot creation. Default is None.
        """
        if self.vlim:
            vlim = self.vlim
        else:
            vlim = PreprocessUtilz.get_ylim(self.features
                                            , feature_name
                                            , [self.group_col, self.channel_col]
                                            , offset=False
                                            , error='se')

        if axes is None:
            fig, axes = plt.subplots(1, self.n_groups
                                     , figsize=(self.n_groups * self.dims[0], self.dims[1] * 0.64)
                                     , constrained_layout=True)
        else:
            fig = np.ravel(axes)[0].figure

        for i, group in enumerate(self.groups):
            group_mask = self.features[self.group_col] == group
            if group_names:
                im = self.plot_topomap(self.features.loc[group_mask]
                                       , feature_name
                                       , ax=axes[i]
                                       , title=group
                                       , vlim=vlim
                                       )
            else:
                im = self.plot_topomap(self.features.loc[group_mask]
                                       , feature_name
                                       , ax=axes[i]
                                       , vlim=vlim
                                       )

        # Add a single colorbar to the right
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label(display_name, fontsize=11)

        return fig

    def plot_topomap_mat(self):
        """
        Creates figure with the topological distribution of the given feature name over the scalp using '.montage'.
        The figure consists of a 2D matrix of subplots: one column per each group, defined using '.group_col',
        and one row for each feature, defined using '.feature_names'.
        """
        n_features = len(self.feature_names)

        if n_features > 1:
            fig, axes = plt.subplots(n_features, self.n_groups,
                                     figsize=(self.n_groups * self.dims[0], n_features * self.dims[1] * 0.64))

            for i, fname in enumerate(self.feature_names):
                self.plot_topomap_row(feature_name=fname,
                                      display_name=self.display_names[i],
                                      group_names=(i == 0),
                                      axes=axes[i])

        else:  ## only 1 feature
            fig = self.plot_topomap_row(feature_name=self.feature_names[0], display_name=self.display_names[0])

        if self.title:
            fig.suptitle(self.title, fontsize=14, y=0.96)
        else:
            fig.suptitle(f'Topological Distributions Between Groups', fontsize=14, y=0.96)

        plt.show()


class StatsUtilz:
    @staticmethod
    def anova(data: pd.DataFrame, feature_name: str, group_col: str = 'clusterID', print_line: bool = True) -> dict:
        """
        One-way ANOVA across clusters for a single numeric feature.
        Reports F, p, df1, df2, eta-squared (η²).
        :param data: a dataframe with the actual numeric data and splitter columns.
        :param feature_name: a name of a column within data on which to perform the analysis.
        :param group_col: a name of a column within data for group splitting (e.g. 'clusterID').
        :param print_line: if True, prints all statistics in a comprehensive line.
        """
        groups = []
        levels = data[group_col].dropna().unique()
        for lev in levels:
            groups.append(data.loc[data[group_col] == lev, feature_name].dropna().values)

        groups = [g for g in groups if len(g) > 0]

        F, p = f_oneway(*groups)

        k = len(groups)
        n = sum(len(g) for g in groups)
        df1 = k - 1
        df2 = n - k

        eta_sq = (F * df1) / (F * df1 + df2) if (F * df1 + df2) > 0 else np.nan

        if print_line:
            print(f"{feature_name}: F({df1},{df2}) = {F:.3f}, p = {p:.6f}, η² = {eta_sq:.3f}")

        return {
            "feature": feature_name,
            "k_groups": k,
            "n_total": n,
            "F": F,
            "p": p,
            "df1": df1,
            "df2": df2,
            "eta_sq": eta_sq
        }

    @staticmethod
    def pairwise_posthoc(data: pd.DataFrame, feature_name: str, group_col: str = 'clusterID',
                         print_line: bool = True) -> str:
        """
        Conducts a tukey posthoc analysis test, assuming a significant one-way ANOVA.
        :param data: a dataframe with the actual numeric data and splitter columns.
        :param feature_name: a name of a column within data on which to perform the analysis.
        :param group_col: a name of a column within data for group splitting (e.g. 'clusterID').
        :param print_line: if True, prints all statistics in a comprehensive line.
        """
        data_c = data[[feature_name, group_col]].dropna().copy()
        posthoc = pairwise_tukeyhsd(endog=data_c[feature_name], groups=data_c[group_col], alpha=0.05)

        summary = posthoc.summary()
        if print_line:
            print(summary)

        return summary

    def chi_squared(data: pd.DataFrame, feature_name: str, group_col: str = 'clusterID', id_col: str = 'Subject',
                    print_line: bool = True) -> dict:
        """
        Chi-square test of independence between group_col (e.g., 'clusterID') and a categorical feature.
        Builds a contingency table using unique counts of id_col per cell (robust to duplicates).
        Reports χ², df, p and Cramer's V.
        :param data: a dataframe with the actual numeric data and splitter columns.
        :param feature_name: a name of a column within data on which to perform the analysis.
        :param group_col: a name of a column within data for group splitting (e.g. 'clusterID').
        :param id_col: a column name within data of unique IDs or keys for each sample (e.g. 'Subject').
        :param print_line: if True, prints all statistics in a comprehensive line.
        """
        ct = (data.groupby([group_col, feature_name])[id_col]
              .nunique()
              .unstack(fill_value=0))

        ct = ct.loc[(ct.sum(axis=1) > 5), (ct.sum(axis=0) > 5)]

        chi2, p, dof, exp = chi2_contingency(ct.values, correction=False)

        n = ct.values.sum()
        r, c = ct.shape
        k = min(r, c)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan

        if print_line:
            print(f"{feature_name}: χ²({dof}) = {chi2:.3f}, p = {p:.6f}, V = {cramers_v:.3f}")

        return {
            "feature": feature_name,
            "table": ct,
            "chi2": chi2,
            "df": dof,
            "p": p,
            "cramers_v": cramers_v,
            "n_total": int(n),
            "shape": (r, c)
        }

    def correct_ps(Ps: np.ndarray | list) -> np.ndarray:
        """
        Performs FDR-BH statistical correction for multiple comparisons.
        :param Ps: a sequence of p-values to correct.
        """
        _, pvals_corrected, _, _ = multipletests(Ps, alpha=0.05, method='fdr_bh')
        return pvals_corrected

    def calc_stats(data: pd.DataFrame, feature_families: dict):
        """
        Calculating statistical test score including p-values and effect-size measures for the given features.
        In addition, computes corrected p-values for multiple comparisons within each feature family using the FDR-BH method.
        """
        all_frames = []
        effect_sizes = []
        for family, features in feature_families.items():
            Ps = []
            for f in features:
                if pd.api.types.is_numeric_dtype(data[f]):
                    res = StatsUtilz.anova(data, f)
                    effect_sizes.append(res['eta_sq'])
                else:
                    res = StatsUtilz.chi_squared(data, group_col='clusterID', feature_name=f)
                    effect_sizes.append(res['cramers_v'])
                Ps.append(res['p'])

            _, corrected, _, _ = multipletests(Ps, alpha=0.05, method='fdr_bh')

            all_frames.append(pd.DataFrame(
                {'Variable': features, 'RawP': Ps, 'CorrectedP': corrected}
            ))

        stats = pd.concat(all_frames)
        stats['EffectSize'] = effect_sizes
        return stats


class ClusterEval:
    @staticmethod
    def get_intrinsic_metrics(data: np.ndarray, models: list, model_names: list[str]) -> pd.DataFrame:
        """
        Get intrinsic clustering metrics to evaluate the model.
        The metrics calculated are: silhouette score, Calniski-Harbasz index (CH) and Davies-Bouldin index (DBI).
        In addition, for the sake of DBSCAN evaluation, the function computes validity scores - defined by
        wether there are more than 1 non-noise labels. For other models, you may ignore that.
        :return: a dataframe with the intrinsic metrics calculated for each model.
        """
        n_models = len(models)
        loo = LeaveOneOut()

        test_shape = (n_models, data.shape[0])
        silhouette_scores = np.zeros(test_shape)
        CH_scores = np.zeros(test_shape)
        DBI_scores = np.zeros(test_shape)
        validity_scores = np.zeros(test_shape)

        for i, model in enumerate(models):
            for train_idx, j in loo.split(data):
                sample = data[train_idx]
                labels = model.fit_predict(sample)

                non_noise_mask = labels != -1
                non_noise_labels = labels[non_noise_mask]
                non_noise_data = sample[non_noise_mask]

                if np.unique(non_noise_labels).size > 1:  # more than 1 cluster
                    prep_sample = PreprocessUtilz._preprocess_pipeline(model, non_noise_data)
                    silhouette_scores[i, j] = silhouette_score(prep_sample, non_noise_labels)
                    CH_scores[i, j] = calinski_harabasz_score(prep_sample, non_noise_labels)
                    DBI_scores[i, j] = davies_bouldin_score(prep_sample, non_noise_labels)
                    validity_scores[i, j] = 1

                else:  # 1 cluster or None
                    silhouette_scores[i, j] = np.nan
                    CH_scores[i, j] = np.nan
                    DBI_scores[i, j] = np.nan

        silhouette_means = np.nanmean(silhouette_scores, axis=1)
        CH_means = np.nanmean(CH_scores, axis=1)
        DBI_means = np.nanmean(DBI_scores, axis=1)
        validity_means = np.nanmean(validity_scores, axis=1)

        silhouette_stds = np.nanstd(silhouette_scores, axis=1)
        CH_stds = np.nanstd(CH_scores, axis=1)
        DBI_stds = np.nanstd(DBI_scores, axis=1)
        validity_stds = np.nanstd(validity_scores, axis=1)

        # Creating a DataFrame with the metrics
        metric_names = ['silhouette_score', 'CH_score', 'DBI_score', 'Validity']
        if model_names is None:
            model_names = list(range(n_models))

        intrinsic_metrics = pd.DataFrame({
            'ModelName': len(metric_names) * model_names
            , 'Metric': np.concatenate([n_models * [mname] for mname in metric_names])
            , 'Mean': np.concatenate([silhouette_means, CH_means, DBI_means, validity_means])
            , 'STD': np.concatenate([silhouette_stds, CH_stds, DBI_stds, validity_stds])
        })

        pivoted = intrinsic_metrics.melt(
            id_vars=['ModelName', 'Metric'],
            var_name='Statistic',
            value_name='Value'
        ).pivot_table(
            index='Metric',
            columns=['ModelName', 'Statistic'],
            values='Value'
        )
        return pivoted

    @staticmethod
    def test_stability(data: np.ndarray, model_path: str, n_trials: int = 100, random_seed: bool = True) -> np.ndarray:
        """
        Test within-model stability between clustering predictions using the adjusted rand index (ARI)
        in a leave-one-out paradigm, with concurrent validation trials.
        :param data: an input numpy array for the model.
        :param model_path: a path to .pkl files with the clustering model.
        This function assumes that the model is a Pipeline object.
        :param n_trials: the number of vaidation trials to perform.
        :param random_seed: if True, randomizes a new random seed for each iteration.
        If False, sets all random seeds to be None.
        """
        model_name = model_path.split(os.sep)[-1].split('.')[0]
        model_1 = joblib.load(model_path)
        model_2 = joblib.load(model_path)

        # Resetting existing random seeds in each step for each model
        if not random_seed:
            PreprocessUtilz.set_random_seeds([model_1, model_2], seed=None)

        n = data.shape[0]
        loo = LeaveOneOut()
        ARIs = np.zeros((n_trials, n))

        # Iterating on subjects in a Leave-One-Out paradigm
        for train_idx, test_idx in tqdm(loo.split(data)):
            sample = data[train_idx]

            # Iterating n_trials times
            for t in range(n_trials):
                # Specifing distinct random seeds for each model
                if random_seed:
                    seed_1 = random.randint(1, 1e6)
                    PreprocessUtilz.set_random_seeds([model_1], seed=seed_1)

                    seed_2 = random.randint(1, 1e6)
                    PreprocessUtilz.set_random_seeds([model_2], seed=seed_2)

                ARIs[t][test_idx] = adjusted_rand_score(model_1.fit_predict(sample), model_2.fit_predict(sample))

        return ARIs

    @staticmethod
    def test_agreement(data: np.ndarray, model_paths: list[str], n_trials: int = 100, random_seed: bool = True) -> dict:
        """
        Test pair-wise model agreement between clustering predictions using the adjusted rand index (ARI)
        in a leave-one-out paradigm, with concurrent validation trials.
        :param data: an input numpy array for the model.
        :param model_paths: a list of paths to .pkl files with the clustering models.
        This function assumes that each model is a Pipeline object.
        :param n_trials: the number of vaidation trials to perform.
        :param random_seed: if True, randomizes a new random seed for each iteration.
        If False, sets all random seeds to be None.
        """
        model_names = [path.split(os.sep)[-1].split('.')[0] for path in model_paths]
        models = [joblib.load(path) for path in model_paths]

        # Resetting existing random seeds in each step for each model
        if not random_seed:
            PreprocessUtilz.set_random_seeds(models, seed=None)

        n = data.shape[0]
        loo = LeaveOneOut()
        model_pairs = list(combinations(model_names, 2))
        pair_names = ["_".join(pair) for pair in model_pairs]
        ARIs = {pair: np.zeros((n_trials, n)) for pair in pair_names}

        # Iterating on subjects in a Leave-One-Out paradigm
        for train_idx, test_idx in tqdm(loo.split(data)):
            sample = data[train_idx]

            # Iterating n_trials times
            for t in range(n_trials):
                if random_seed:
                    seed = random.randint(1, 1e6)
                    PreprocessUtilz.set_random_seeds(models, seed=seed)

                labels = {name: None for name in model_names}

                # Predicting cluster labels for each model
                for i, m in enumerate(models):
                    name = model_names[i]
                    labels[name] = m.fit_predict(sample)

                # Computing pair-wise ARI
                for i, (a, b) in enumerate(model_pairs):
                    ARIs[pair_names[i]][t][test_idx] += adjusted_rand_score(labels[a], labels[b])

        return ARIs

    @staticmethod
    def radar_plot(data: pd.DataFrame, metrics: list[str], group_col: str = 'clusterID', title: str = None,
                   normalize: str = None, ylabel: str = None, figsize: (int, int) = (5, 6), yticks: np.ndarray = None,
                   metric_names: list[str] = None, show_err: bool = True, spin=np.pi / 2):
        """
        A general function to create a radar plot including several feature.
        :param data: a dataframe with the feature values.
        :param metrics: a list of column names within data correspoding to the relevant features for the plot.
        :param group_col: the name of the group column within clusters. Default is 'clusterID'.
        :param title: a title for the figure. Default is None.
        :param normalize: if True, normalizing the values within each features using Z-score before creating the plot.
        :param ylabel: a label for the y-axis. Default is None.
        :param figsize: the figure size. Default is (5, 6).
        :param yticks: an array of specific ticks for the y-axis. Default is None.
        :param metric_names: a list of names correspoding to metrics to display in the plot.
        :param show_err: if True, draws the standard error bar for each group. Default is True.
        :parm spin: a rotation factor for the plot in radians. Default is np.pi / 2.
        """
        groupby = data.groupby([group_col])[metrics]
        means = groupby.mean().sort_index()
        ses = groupby.sem().sort_index()

        means, ses = PreprocessUtilz.normalize_group_means(means=means, ses=ses, mode=normalize)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate([angles, angles[:1]])

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)  # << this line makes it a radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics if metric_names is None else metric_names, fontsize=14)
        ax.tick_params(axis="x", pad=15)  # push feature labels outward a bit

        if yticks is not None:
            ax.set_yticks(yticks)

        ax.yaxis.grid(True, alpha=0.6)

        if ylabel is None:
            ylabel = 'Value' if normalize is None else normalize
        ax.set_ylabel(ylabel, labelpad=50, fontsize=11)
        ax.yaxis.set_label_coords(0.5, -0.2)
        ax.yaxis.label.set_rotation(0)

        for cid, row in means.iterrows():
            vals = row.values
            curve = np.concatenate([vals, vals[:1]])
            ax.plot(angles, curve, linewidth=2, label=cid)

            if show_err:
                se = ses.loc[cid].values
                lower = np.concatenate([vals - se, [vals[0] - se[0]]])
                upper = np.concatenate([vals + se, [vals[0] + se[0]]])
                ax.fill_between(angles, lower, upper, alpha=0.15)

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), title="Cluster")
        ax.set_theta_offset(spin)
        ax.set_theta_direction(-1)
        ax.set_title(title, pad=16, fontsize=14, loc='center')
        plt.show()

    @staticmethod
    def plot_multiple_features(features: pd.DataFrame, clusters: pd.DataFrame, raw_feature_names: list[str],
                               disp_feature_names: list[str] = None,
                               xlabel: str = 'Feature Name', ylabel: str = 'Value', cluster_name: str = 'K-Means',
                               title: str = 'Feature Values',
                               plot_type: str = 'line', ylim=None, group_col: str = 'clusterID'):
        """
        A general function to plot inter-group differences of several features, between behavioral conditions.
        :param features: a dataframe with the feature values for each subject.
        :param clusters: a dataframe with the cluster assignment for each subject.
        :param raw_feature_names: a list of feature names corresponding to colums in features.
        :param disp_feature_names: a list of feature names for display in the figure, corresponding to raw_feature_names.
            If None, raw_feature_names are used.
        :param xlabel: a label for the x-axis. Default is 'Feature Name'.
        :param ylabel: a label for the y-axis. Default is 'Value'.
        :param cluster_name: the name of the clustering model, or otherwise grouping scheme for the subjects (e.g. K-Means).
        :param title: a title prefix for the figure. Full title will be '{title} between {cluster_name} Clusters Across Behavioral States'.
        :param plot_type: a name of a plot type, should be one of ['bar', 'line'].
        :param ylim: a tuple of limits for the y-axis in the figure. If None, limits are set automatically via seaborn.
        :param group_col: the name of the group column within clusters. Default is 'clusterID'.
        """
        if disp_feature_names is None:
            disp_feature_names = raw_feature_names

        selected_features = features.loc[features['FeatureName'].isin(raw_feature_names)]
        agged_features = selected_features.groupby(['Subject', 'Condition', 'FeatureName'])[
            ['RawValue']].mean().reset_index()

        agged_features = pd.merge(agged_features, clusters, how='inner', on='Subject')

        agged_features['FeatureName'] = pd.Categorical(
            agged_features['FeatureName'],
            categories=raw_feature_names,
            ordered=True
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if plot_type == 'line':
            sns.lineplot(agged_features.loc[agged_features['Condition'] == 'sit'],
                         x='FeatureName', y='RawValue', hue=group_col, ax=axes[0])

            sns.lineplot(agged_features.loc[agged_features['Condition'] == 'walk'],
                         x='FeatureName', y='RawValue', hue=group_col, ax=axes[1])
        elif plot_type == 'bar':
            sns.barplot(agged_features.loc[agged_features['Condition'] == 'sit'],
                        x='FeatureName', y='RawValue', hue=group_col, ax=axes[0], capsize=0.2)

            sns.barplot(agged_features.loc[agged_features['Condition'] == 'walk'],
                        x='FeatureName', y='RawValue', hue=group_col, ax=axes[1], capsize=0.2)

        axes[0].set_title('Resting-State')
        axes[1].set_title('Active-Walking')

        for i in range(2):
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_xticklabels(disp_feature_names)

            if ylim is not None:
                axes[i].set_ylim(ylim)

        plt.suptitle(f'{title} between {cluster_name} Clusters Across Behavioral States', fontsize=14, y=1.02)
        plt.show()

    @staticmethod
    def plot_single_feature(features: pd.DataFrame, clusters: pd.DataFrame, feature_name: str,
                            ylabel: str = 'Value', cluster_name: str = 'K-Means', title: str = 'Feature Values',
                            plot_type: str = 'line', ylim=None, group_col: str = 'clusterID'):
        """
        A general function to plot inter-group differences of a single feature, between behavioral conditions.
        :param features: a dataframe with the feature values for each subject.
        :param clusters: a dataframe with the cluster assignment for each subject.
        :param feature_name: a name of a feature corresponding to colums in features.
        :param ylabel: a label for the y-axis. Default is 'Value'.
        :param cluster_name: the name of the clustering model, or otherwise grouping scheme for the subjects (e.g. K-Means).
        :param title: a title prefix for the figure. Full title will be '{title} between {cluster_name} Clusters Across Behavioral States'.
        :param plot_type: a name of a plot type, should be one of ['bar', 'line'].
        :param ylim: a tuple of limits for the y-axis in the figure. If None, limits are set automatically via seaborn.
        :param group_col: the name of the group column within clusters. Default is 'clusterID'.
        """
        selected_features = features.loc[features['FeatureName'] == feature_name]
        agged_features = selected_features.groupby(['Subject', 'Condition'])[['RawValue']].mean().reset_index()
        agged_features = pd.merge(agged_features, clusters, how='inner', on='Subject')

        if plot_type == 'line':
            ax = sns.lineplot(agged_features, x='Condition', y='RawValue', hue=group_col)

        elif plot_type == 'bar':
            ax = sns.barplot(agged_features, x='Condition', y='RawValue', hue=group_col, capsize=0.2)

        plt.ylabel(ylabel)

        if ylim is not None:
            plt.ylim(ylim)

        ax.set_xticklabels(['Resting-State', 'Active-Walking'])

        plt.title(f'{title} between {cluster_name} Clusters Across Behavioral States', fontsize=12, y=1.02)
        plt.show()


class ClusterReport:
    def __init__(self, models: list[GaussianMixture], model_names: list[str], PD_data: np.ndarray,
                 PD_subjects: np.ndarray
                 , HC_data: np.ndarray, cl_data_path: str, cl_fnames: list[str], cl_VGNG_names: list[str] = None
                 , k: int = 5, resample_rate: float = 0.8):
        """
        An object for fast evaluation of Parkinson's disease clustering models using conventional performance indices and clinical assessments.
        :param models: a list of unfitted sklearn clustering models which one wish to evaluate.
        :param model_names: labels corresponding to the models, for referral to results.
        :param PD_data: a list of feature matrices of the Parkinson's Disease (PD) subjects, for each model.
        :param PD_subjects: an array of the subjectIDs for the Parkinson's Disease (PD) subjects.
        :param HC_data: a list of feature matrices of the Parkinson's Disease (PD) subjects, for each model.
        :param cl_data_path: a path to the .xlsx file of the clinical data.
        :param cl_fnames: a list of feature names from the clinical data to include in evaluation (e.g. LEDD, UPDRS).
        :param cl_VGNG_names: a list of VGNG-related feature names from the clinical data to include in evaluation.
        """
        self.models = models
        self.model_names = model_names
        self.n_models = len(models)

        self.PD_data = PD_data
        self.PD_subjects = PD_subjects
        self.HC_data = HC_data

        self.cl_data: pd.DataFrame = IOUtilz.read_clinical_features(cl_data_path)
        self.cl_fnames = cl_fnames
        self.cl_VGNG_names = cl_VGNG_names

        self.loo = LeaveOneOut()

    def set_models(self, models: list, model_names: list[str]):
        """
        Set the model attributes for object reuse purposes.
        """
        self.models = models
        self.model_names = model_names
        self.n_models = len(models)

    def set_data(self, PD_data: np.ndarray, HC_data: np.ndarray):
        """
        Set the data attributes for object reuse purposes.
        """
        self.PD_data = PD_data
        self.HC_data = HC_data

    def _fit_models(self):
        """
        Fit all models to all Parkinson's Disease (PD) data.
        """
        for i in range(self.n_models):
            self.models[i].fit(self.PD_data)

    def _get_intrinsic_metrics(self, data: np.ndarray) -> pd.DataFrame:
        """
        Get intrinsic clustering metrics to evaluate the model.
        The metrics calculated are: silhouette score, Calniski-Harbasz index (CH) and Davies-Bouldin index (DBI).
        In addition, for the sake of DBSCAN evaluation, the function computes validity scores - defined by
        wether there are more than 1 non-noise labels. For other models, you may ignore that.
        :return: a dataframe with the intrinsic metrics calculated for each model.
        """
        return ClusterEval.get_intrinsic_metrics(data=self.PD_data,
                                                 models=self.models,
                                                 model_names=self.model_names)

    def _get_clinical_means(self, fnames: list[str]) -> pd.DataFrame:
        """
        Get the means of the clinical measures for each cluster in each model.
        Assumes models are already fitted to the data.
        :param fnames: a list of feature names from the clinical data to include in calculation.
        :return: a dataframe with the mean of each clinical measure in fnames, per cluster and model.
        """
        # Slicing only PD subjects found in subjects
        PD_subject_mask = self.cl_data['Subject'].isin(self.PD_subjects)
        cl_features = self.cl_data.loc[PD_subject_mask, fnames]
        cl_feature_mat = cl_features.values
        cl_feature_mat[np.isnan(cl_feature_mat)] = 0

        # Slicing only PD subjects found in clinical data
        subject_mask = np.where(np.isin(self.PD_subjects, self.cl_data['Subject'].values))[0]

        metrics_per_model = []
        N = subject_mask.size
        for i in range(self.n_models):
            data = self.PD_data[subject_mask]
            cl_metrics = pd.DataFrame(cl_feature_mat, columns=fnames)
            cl_metrics['ClusterID'] = self.models[i].fit_predict(data)
            cl_metrics['Model'] = self.model_names[i]
            metrics_per_model.append(cl_metrics)

        # Adding healthy controls for reference
        HC_subject_mask = self.cl_data['Subject'].str.startswith('HC')
        cl_features = self.cl_data.loc[HC_subject_mask, fnames]
        cl_feature_mat = cl_features.values
        cl_feature_mat[np.isnan(cl_feature_mat)] = 0
        for i in range(self.n_models):
            cl_metrics = pd.DataFrame(cl_feature_mat, columns=fnames)
            cl_metrics['ClusterID'] = 'HC'
            cl_metrics['Model'] = self.model_names[i]
            metrics_per_model.append(cl_metrics)

        all_clinical_metrics = pd.concat(metrics_per_model)
        return all_clinical_metrics.groupby(['Model', 'ClusterID']).mean().reset_index(drop=False)

    def _visualize_metrics(self, metrics: pd.DataFrame):
        """
        Visualize metrics in table in heatmap-figure forms, grouped by model.
        :param metrics: a dataframe with a Model CLusterID columns, and other numerical columns to be visualized.
        """
        select_cols = [col for col in metrics.columns if col != 'Model']

        if self.n_models == 1:  # Only one model to evaluate
            model_cl_data = metrics[select_cols].set_index('ClusterID')
            norm_cl_data = model_cl_data.apply(PreprocessUtilz.normalize_col, axis=0)
            plt.figure(figsize=(8, 3))
            sns.heatmap(norm_cl_data.T, cmap='coolwarm', cbar_kws={'label': 'Normalized Value'},
                        annot=model_cl_data.round(3).T)
            plt.title(self.model_names[0].title())

        else:  # Several models to evaluate
            fig, axes = plt.subplots(1, self.n_models, figsize=(self.n_models * 8, 3))
            for i in range(self.n_models):
                name = self.model_names[i]
                model_cl_data = metrics.loc[metrics['Model'] == name, select_cols].set_index('ClusterID')
                norm_cl_data = model_cl_data.apply(PreprocessUtilz.normalize_col, axis=0)
                sns.heatmap(norm_cl_data.T, ax=axes[i], cmap='coolwarm', cbar_kws={'label': 'Normalized Value'},
                            annot=model_cl_data.round(3).T)
                axes[i].set_title(name.title())

            plt.suptitle('Metrics Between Clusters')

        plt.show()
        plt.close()

    @staticmethod
    def _UMAP_project(data) -> np.ndarray:
        """
        Project the given data into a lower dimension using the UMAP algorithm.
        :param data: a dataset to evaluate models on.
        """
        reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=SEED)
        return reducer.fit_transform(data)

    def _UMAP_visualize(self):
        """
        Creates a scatter plot for each model where UMAP-projected data points are colored by hard-clustering labels.
        Assumes models are already fitted to the data.
        :param data: a dataset to evaluate models on.
        """
        plt.figure()
        if self.n_models == 1:
            prepped = PreprocessUtilz._preprocess_pipeline(self.models[0], self.PD_data)
            UMAP_proj = ClusterReport._UMAP_project(prepped)
            x = UMAP_proj[:, 0]
            y = UMAP_proj[:, 1]
            labels = self.models[0].fit_predict(self.PD_data)
            scatter = plt.scatter(x=x, y=y, c=labels)
            plt.title('UMAP Projected Clusters (PD only)')
            plt.xlabel('UMAP-comp1')
            plt.ylabel('UMAP-comp2')
            legend = plt.legend(*scatter.legend_elements(), title="ClusterID")
            plt.gca().add_artist(legend)
        else:
            fig, axes = plt.subplots(1, self.n_models, figsize=(self.n_models * 6, 5))
            for i, model in enumerate(self.models):
                prepped = PreprocessUtilz._preprocess_pipeline(model, self.PD_data)
                UMAP_proj = ClusterReport._UMAP_project(prepped)
                x = UMAP_proj[:, 0]
                y = UMAP_proj[:, 1]
                labels = model.fit_predict(self.PD_data)
                scatter = axes[i].scatter(x=x, y=y, c=labels)
                axes[i].set_title(self.model_names[i].title(), fontsize=14)
                axes[i].set_xlabel('UMAP-comp1')
                axes[i].set_ylabel('UMAP-comp2')
                legend = axes[i].legend(*scatter.legend_elements(), title="ClusterID")
                axes[i].add_artist(legend)

            plt.suptitle('UMAP Projected Clusters (PD only)', fontsize=18)

        plt.show()
        plt.close()

    def report(self, include_clinical: bool = True):
        print(
            "======================================== Model ClusterEval Report ========================================\n")

        print(
            "--------------------------------------------- General Metrics --------------------------------------------\n")
        intrinsic_metrics = self._get_intrinsic_metrics(self.PD_data)
        print(intrinsic_metrics, '\n')

        print(
            "-------------------------------------- Low Dimensional Visualization -------------------------------------\n")
        self._UMAP_visualize()

        if include_clinical:
            print(
                "-------------------------------------- Clinical Metrics Differences -------------------------------------\n")
            clinical_metrics = self._get_clinical_means(self.cl_fnames)
            self._visualize_metrics(clinical_metrics)

            if self.cl_VGNG_names:
                print(
                    "-------------------------------------- VGNG-related Metrics Differences -------------------------------------\n")
                VGNG_metrics = self._get_clinical_means(self.cl_VGNG_names)
                self._visualize_metrics(VGNG_metrics)


class FeatureImportance:
    def __init__(self, features_metadata: pd.DataFrame, n_trials: int, model_path: str, data: np.ndarray):
        """
        :param features_metadata:
        :param n_trials:
        :param model_path:
        :param data:
        """
        self.features_metadata = features_metadata
        self.n_trials = n_trials
        self.data = data

        self.base_model = joblib.load(model_path)
        self.dr_name = self.base_model.steps[-2][0]
        self.cluster_name = self.base_model.steps[-1][0]

    def create_splits(self, split_conditions: bool = True, split_regions: bool = False):
        """
        Creates data splits for feature importance analysis (via calc_feature_importance).
        :param split_conditions: If True, splits data between behavioral conditions. Deafult is True.
        :param split_conditions: If True, splits data between anatomical regions. Default is False.
        :return splits: a tuple (ex_ids, keep_ids) with the indices of the corresponding columns in the data to exlucde and keep.
        :return labels: a corresponding label for each split (feature_name, condition, region).
        If data was not split by conditions or regions, then condition | region = 'ALL' accordingly.
        """
        splits, labels = [], []

        features = self.features_metadata['FeatureName'].unique()
        conditions = self.features_metadata['Condition'].unique() if split_conditions else np.array(['ALL'])
        regions = self.features_metadata['Region'].unique() if split_regions else np.array(['ALL'])

        for fname in features:
            fmask = (self.features_metadata['FeatureName'] == fname)
            for cond in conditions:
                cmask = (self.features_metadata['Condition'] == cond)
                for reg in regions:
                    rmask = (self.features_metadata['Region'] == reg)

                    if split_conditions and split_regions:
                        mask = fmask & cmask & rmask
                    elif split_conditions:
                        mask = fmask & cmask
                    elif split_regions:
                        mask = fmask & rmask
                    else:
                        mask = fmask

                    ex_idx = np.where(mask)[0]
                    keep_idx = np.where(~mask)[0]
                    splits.append((ex_idx, keep_idx))
                    labels.append((fname, cond, reg))  # ALWAYS 3-tuple

        return splits, labels

    @staticmethod
    def fit_predict(model, X: np.ndarray, fit: bool = True):
        """
        Fits and predicts a clustering model on the input data (X), and returns the cluster labels and silhouette score.
        :param model: a clustering model Pipeline object with DR (-2) and clustering (-1) steps.
        :param X: the data matrix to predict.
        :param fit: If True, fits the model on X, then predicts the labels on X. If False, only performs label prediction.
        """
        if fit:
            labels = model.fit_predict(X)
        else:
            labels = model.predict(X)

        embed = model[:-1].transform(X)
        non_noise = labels != -1
        n_clusters = len(np.unique(labels[non_noise]))
        if n_clusters > 1 and embed.shape[0] > n_clusters:
            sil = silhouette_score(embed[non_noise], labels[non_noise])
        else:
            sil = np.nan

        return labels, sil

    def calc_feature_importance(self, split_conditions: bool = True, split_regions: bool = False):
        """
        Calculates feature importance on the input data using a permutation importance approach.
        :param split_conditions: If True, splits data between behavioral conditions. Deafult is True.
        :param split_conditions: If True, splits data between anatomical regions. Default is False.
        :return: a dataframe with the silhouette drop and ARI for each permuted data split, within each trial.
        """
        splits, labels = self.create_splits(split_conditions, split_regions)

        shape = (len(splits), self.n_trials)
        ARI = np.zeros(shape, dtype=float)
        silhouette = np.zeros(shape, dtype=float)

        rng = np.random.default_rng(42)

        for j in tqdm(range(self.n_trials)):
            seed = int(rng.integers(0, 1_000_000))

            # ---- FULL baseline once per trial ----
            if self.cluster_name == 'DBSCAN':
                seed_params = {f"{dr_name}__random_state": seed}
            else:
                seed_params = {
                    f"{self.dr_name}__random_state": seed,
                    f"{self.cluster_name}__random_state": seed
                }
            raw_model = clone(self.base_model).set_params(**seed_params)

            raw_labels, raw_sil = FeatureImportance.fit_predict(model=raw_model, X=self.data)

            # ---- Permutation ----
            for i, (ex_idx, keep_idx) in enumerate(splits):
                Xp = self.data.copy()
                for c in np.atleast_1d(ex_idx):
                    Xp[:, c] = rng.permutation(Xp[:, c])

                perm_labels, perm_sil = FeatureImportance.fit_predict(model=raw_model,
                                                                      X=Xp,
                                                                      fit=self.cluster_name == 'DBSCAN')

                ARI[i, j] = adjusted_rand_score(raw_labels, perm_labels)
                silhouette[i, j] = (raw_sil - perm_sil) if (not np.isnan(raw_sil) and not np.isnan(perm_sil)) else 0.0

        labels_arr = np.array(labels, dtype=object)  # shape (M, 2)
        feat_series = np.repeat(labels_arr[:, 0], self.n_trials)  # FeatureName repeated
        cond_series = np.repeat(labels_arr[:, 1], self.n_trials)  # Condition repeated
        reg_series = np.repeat(labels_arr[:, 2], self.n_trials)  # region repeated
        trials = np.tile(np.arange(1, self.n_trials + 1), len(splits))

        return pd.DataFrame({
            "FeatureName": feat_series,
            "Condition": cond_series,
            "Region": reg_series,
            "Trial": trials,
            "ARI": ARI.reshape(-1),
            "SilhouetteDrop": silhouette.reshape(-1)
        })

    @staticmethod
    def plot_feature_importance(importance: pd.DataFrame, order=None, xlabels=None):
        """
        Plots feature importance data that were pre-calculated.
        :param importance: a dataframe with the feature importance results.
        :param order: a list representing the ordr of features for display.
        :param xlables: an alternative list of xlabels, corresponding to each feature in the given order.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.barplot(importance, hue='Condition', x='FeatureName', y='ARI',
                    ax=axes[0], capsize=.2, errorbar='se', order=order)
        axes[0].set_title('Adjusted Rand Index (ARI) between Raw and Permuted Data')
        axes[0].set_ylim(0.7, 0.95)
        axes[0].set_xlabel('Feature Name')
        if xlabels is not None:
            axes[0].set_xticklabels(xlabels)

        axes[0].legend(loc='upper right')

        sns.barplot(importance, hue='Condition', x='FeatureName', y='SilhouetteDrop',
                    ax=axes[1], capsize=.2, errorbar='se', order=order)
        axes[1].set_title('Drop in Silhouette Score After Permutation')
        axes[1].set_xlabel('Feature Name')
        axes[1].set_ylabel('$\Delta$ Silhouette')
        if xlabels is not None:
            axes[1].set_xticklabels(xlabels)

        axes[1].legend(loc='upper right')

        plt.suptitle('Permutation Feature Importance between Conditions', fontsize=16, y=1.02)
        plt.show()
