import os
import scipy
import warnings
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from EntropyHub import PermEn
from antropy import higuchi_fd, lziv_complexity
from sklearn.preprocessing import StandardScaler
from specparam import SpectralGroupModel, Bands
from mne.io.eeglab.eeglab import EpochsEEGLAB

from EEGLABIO import EEGLABIO


RES_DIR = os.path.join('..', 'results')


class FEUtilz:
    
    ELECTRODE_NAME_CONVERSION_MAP = {
        "E1": "F10", "E2": "AF4", "E3": "F2", "E4": "FCz"
        , "E5": "Fp2", "E6": "Fz", "E7": "FC1", "E8": "AFz"
        , "E9": "F1", "E10": "Fp1", "E11": "AF3", "E12": "F3"
        , "E13": "F5", "E14": "FC5", "E15": "FC3", "E16": "C1"
        , "E17": "F9", "E18": "F7", "E19": "FT7", "E20": "C3"
        , "E21": "CP1", "E22": "C5", "E23": "T9", "E24": "T7"
        , "E25": "TP7", "E26": "CP5", "E27": "P5", "E28": "P3"
        , "E29": "TP9", "E30": "P7", "E31": "P1", "E32": "P9"
        , "E33": "PO3", "E34": "Pz", "E35": "O1", "E36": "POz"
        , "E37": "Oz", "E38": "PO4", "E39": "O2", "E40": "P2"
        , "E41": "CP2", "E42": "P4", "E43": "P10", "E44": "P8"
        , "E45": "P6", "E46": "CP6", "E47": "TP10", "E48": "TP8"
        , "E49": "C6", "E50": "C4", "E51": "C2", "E52": "T8"
        , "E53": "FC4", "E54": "FC2", "E55": "T10", "E56": "FT8"
        , "E57": "FC6", "E58": "F8", "E59": "F6", "E60": "F4"
        , "E61": "EOG1", "E62": "EOG2", "E63": "EOG3", "E64": "EOG4"
    }
    
    ELECTRODE_REGION_MAP = {
        'Fp2': 'FrontalRight', 'AF4': 'FrontalRight', 'F10': 'FrontalRight', 'F2': 'FrontalRight'
        , 'F4': 'FrontalRight', 'F6': 'FrontalRight', 'F8': 'FrontalRight', 'FC2': 'FrontalRight'
        , 'FC4': 'FrontalRight', 'FC6': 'FrontalRight', 'FT8': 'FrontalRight', 'AFz': 'FrontalMidline'
        , 'Fz': 'FrontalMidline', 'FCz': 'FrontalMidline', 'F1': 'FrontalLeft', 'F9': 'FrontalLeft'
        , 'Fp1': 'FrontalLeft', 'FC5': 'FrontalLeft', 'AF3': 'FrontalLeft', 'F7': 'FrontalLeft'
        , 'F5': 'FrontalLeft', 'F3': 'FrontalLeft', 'FT7': 'FrontalLeft', 'FC3': 'FrontalLeft'
        , 'FC1': 'FrontalLeft', 'CP2': 'ParietalRight', 'CP6': 'ParietalRight', 'P2': 'ParietalRight'
        , 'P4': 'ParietalRight', 'P6': 'ParietalRight', 'P8': 'ParietalRight', 'P10': 'ParietalRight'
        , 'PO4': 'ParietalRight', 'Pz': 'ParietalMidline', 'POz': 'ParietalMidline', 'CP1': 'ParietalLeft'
        , 'CP5': 'ParietalLeft', 'P1': 'ParietalLeft', 'P3': 'ParietalLeft', 'P5': 'ParietalLeft'
        , 'P7': 'ParietalLeft', 'P9': 'ParietalLeft', 'PO3': 'ParietalLeft', 'T8': 'TemporalRight'
        , 'TP8': 'TemporalRight', 'T10': 'TemporalRight', 'TP10': 'TemporalRight', 'T7': 'TemporalLeft'
        , 'TP7': 'TemporalLeft', 'T9': 'TemporalLeft', 'TP9': 'TemporalLeft', 'O2': 'OccipitalRight'
        , 'Oz': 'OccipitalMidline', 'O1': 'OccipitalLeft', 'C2': 'CentralRight', 'C4': 'CentralRight'
        , 'C6': 'CentralRight', 'C5': 'CentralLeft', 'C3': 'CentralLeft', 'C1': 'CentralLeft'
        }
    
    @staticmethod
    def build_feature_df(electrodes: list[str], feature_name: str, feature_category: str, values) -> pd.DataFrame:
        """
        Building a dataframe with the specified fields.
        :param electrodes: a list of the names of electrodes corresponding to the values.
        :param feature_name: the name of the feature itself for the feature table.
        :param feature_category: the name of the feature category for the feature table.
        :param values: an array-like of values.
        :return: A dataframe with the specified fields.
        """
        table = pd.DataFrame()
        table['Electrode'] = pd.Series(electrodes)
        table['FeatureName'] = feature_name
        table['FeatureCategory'] = feature_category
        table['RawValue'] = pd.Series(values)
        return table

    @staticmethod
    def average_epochs(epochs_data: np.ndarray) -> np.ndarray:
        """
        Calculate the averaged waveform across all epochs within a certain condition and subject.
        :param epochs_data: a 3-dimensional array (epochs, electrodes, samples) of the raw EEG data.
        :return: a 2-dimensional array (electrodes, samples) of the averaged EEG waveform across epochs.
        """
        return epochs_data.mean(axis=0)
    
    @staticmethod
    def slice_epochs_2d(epochs_data: np.ndarray, sampling_rate: int, start_time: int, end_time: int, time_offset: int, electrodes_map: list[str] = None
                        , electrode_names: list[str] = None) -> np.ndarray:
        """
        Slicing epoch to a given time window (mainly for ERP analysis).
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the EEG waveform.
        :param sampling_rate: the sampling rate of the signal in Hz.
        :param start_time: an integer specifying the starting point of the time window in milliseconds.
        :param end_time: an integer specifying the ending point of the time window in milliseconds.
        :param time_offset: the duration of the baseline recordings in milliseconds.
        :param electrodes_map: the list of all electrodes names included in the EEG data tensor.
        :param electrode_indices: a list of names of relevant electrodes in the epochs_data.
        :return sliced: a 2-dimensional array (electrodes, samples) where all the samples are within the time window, from the given electrodes.
        :return updated_electrode_map: an updated version of the given electrodes_map, including only the given electrode_names.
        """
        if (start_time < np.inf) & (end_time < np.inf):
            temporal_resolution = 1000 / sampling_rate
            start_index = int((start_time + time_offset) / temporal_resolution)
            end_index = int((end_time + time_offset) / temporal_resolution)
            
            if electrode_names:
                electrodes_indices, updated_electrode_map = FEUtilz.electrode_name_translate(electrode_names, electrodes_map)
                sliced = epochs_data[electrodes_indices, start_index:end_index]
            else:
                sliced = epochs_data[:, start_index:end_index]
                updated_electrode_map = electrodes_map
        
        else:
            if electrode_names:
                electrodes_indices, updated_electrode_map = FEUtilz.electrode_name_translate(electrode_names, electrodes_map)
                sliced = epochs_data[electrodes_indices, :]
            else:
                sliced = epochs_data
                updated_electrode_map = electrodes_map
            
        return sliced, updated_electrode_map
    
    @staticmethod
    def slice_epochs_3d(epochs_data: np.ndarray, sampling_rate: int, start_time: int, end_time: int, time_offset: int, electrodes_map: list[str] = None
                        , electrode_names: list[str] = None) -> np.ndarray:
        """
        Slicing epoch to a given time window (mainly for ERP analysis).
        :param epochs_data: a 3-dimensional array (electrodes, samples) of the EEG waveform.
        :param sampling_rate: the sampling rate of the signal in Hz.
        :param start_time: an integer specifying the starting point of the time window in milliseconds.
        :param end_time: an integer specifying the ending point of the time window in milliseconds.
        :param time_offset: the duration of the baseline recordings in milliseconds.
        :param electrodes_map: the list of all electrodes names included in the EEG data tensor.
        :param electrode_indices: a list of names of relevant electrodes in the epochs_data.
        :return sliced: a 3-dimensional array (electrodes, samples) where all the samples are within the time window, from the given electrodes.
        :return updated_electrode_map: an updated version of the given electrodes_map, including only the given electrode_names.
        """
        if (start_time < np.inf) & (end_time < np.inf):
            temporal_resolution = 1000 / sampling_rate
            start_index = int((start_time + time_offset) / temporal_resolution)
            end_index = int((end_time + time_offset) / temporal_resolution)
            
            if electrode_names:
                electrodes_indices, updated_electrode_map = FEUtilz.electrode_name_translate(electrode_names, electrodes_map)
                sliced = epochs_data[:, electrodes_indices, start_index:end_index]
            else:
                sliced = epochs_data[:, :, start_index:end_index]
                updated_electrode_map = electrodes_map
        
        else:
            if electrode_names:
                electrodes_indices, updated_electrode_map = FEUtilz.electrode_name_translate(electrode_names, electrodes_map)
                sliced = epochs_data[:, electrodes_indices, :]
            else:
                sliced = epochs_data
                updated_electrode_map = electrodes_map
            
        return sliced, updated_electrode_map
    
    @staticmethod    
    def electrode_name_translate(electrode_names: list[str], electrodes_map: list) -> list[int]:
        """
        Translates electrode names from a specific subject to the corresponding indices in the EEG data tensor.
        :param electrode_name: a list of electrode names (e.g. 'Cz').
        :param electrodes_map: the list of all electrodes included in the EEG data tensor.
        :return electrode_indices: a list of integers representing the index of the electrode within the EEG data tensor.
        If the electrode_name is not found within the electrodes_map, the integer will be -1.
        :return updated_electrode_map: an updated version of the given electrodes_map, including only the given electrode_names.
        """
        electrodes_dict = {name: idx for idx, name in enumerate(electrodes_map)}
    
        electrode_indices = []
        for el_name in electrode_names:
            electrode_indices.append(electrodes_dict.get(el_name, -1))
        
        updated_electrode_map = [electrodes_map[i] for i in electrode_indices if i != -1]
        electrode_indices = [i for i in electrode_indices if i != -1]
        
        return electrode_indices, updated_electrode_map
    
    @staticmethod
    def baseline_correct(epochs_data: np.ndarray, sampling_rate: int, time_offset: int) -> np.ndarray:
        """
        Subtract the mean of the baseline activity from an epoch's entire waveform, across all epochs.
        :param epochs_data: a 3-dimensional array (trials, electrodes, samples) of the EEG waveform.
        :param sampling_rate: the sampling rate of the signal in Hz.
        :param time_offset: the duration of the baseline recordings in milliseconds.
        :return: a 3-dimensional array (trials, electrodes, samples) of the corrected EEG waveform.
        """
        baseline, _ = FEUtilz.slice_epochs_3d(epochs_data, sampling_rate, start_time=(-1)*time_offset, end_time=0, time_offset=time_offset)
        
        baseline_means = baseline.mean(axis=2)
        return epochs_data - baseline_means[:, :, np.newaxis]
    
    @staticmethod
    def convert_electrode_names(electrodes_map: list[str]):
        """
        Convert electorde names from E1-E64 to Fz, Cz, Pz... (10-20 setting).
        :param electrodes_map: the list of electrode names to convert.
        """
        for i in range(len(electrodes_map)):
            curr_elec_name = electrodes_map[i]
            electrodes_map[i] = FEUtilz.ELECTRODE_NAME_CONVERSION_MAP.get(curr_elec_name, curr_elec_name)
            
    @staticmethod
    def normalize(feature_table: pd.DataFrame):
        """
        Normalize values in feature_table into a new 'NormalValue' column. Calculation is performed inplace.
        Normalization is done via applying StandardScalar() transformation to each FeatureName.
        :param feature_table: a dataframe representing the feature table, including the fields: FeatureName, RawValue.
        """
        feature_names = feature_table['FeatureName'].unique()
        for fname in feature_names:
            fmask = feature_table['FeatureName'] == fname
            fvals_raw = feature_table.loc[fmask, ['RawValue']].values
            feature_table.loc[fmask, ['NormalValue']] = StandardScaler().fit_transform(fvals_raw)
    
    @staticmethod
    def store_table(table: pd.DataFrame, file_name: str, results_dir: str = RES_DIR):
        """
        Stores table to a .csv file.
        :param table: a dataframe.
        :param file_name: the desired name for the .csv file.
        :parma results_dir: a path to the results directory, in which to store the table.
        """
        timestamp = dt.datetime.now().strftime('%d_%m_%Y_%H_%M')
        print(f'saved file at {file_name}_{timestamp}.csv')
        table.to_csv(os.path.join(results_dir, f'{file_name}_{timestamp}.csv'))
        
    @staticmethod
    def get_feature_matrix(feature_table: pd.DataFrame, normalize: bool = False, agg_regions: bool = False
                           , get_diffs: bool = False) -> np.ndarray:
        """
        Extract a feature matrix from a feature table calculated by get_feature_table.
        The rows of the matrix correspond to subjects, and the columns are corresponding to features, 
        where each column represents a unique combination of 'FeatureName', 'Condition' and 'Sensor'.
        'Sensor' can correspond to an electrode or to the mean across several electrodes in a specific brain region.
        :param feature_table: the feature table to extract features from.
        :param normalize: if True gets normalized values, otherwise gets raw values.
        :param agg_regions: if True, returns the mean feature value for each brain region.
        Otherwise returns the mean feature value for each electrode. Default is False.
        :param get_diffs: if True, return the absolute difference between conditions as the feature value.
        Otherwise, returns the raw value for each condition.
        :return feature_mat: the feature matrix.
        :return subjects: a vector of subject names corresponding to the feature matrix.
        :return feature_names: an array representing the feature name for each column in the matrix.
        The array contains 3 columns - 'FeatureName', 'Condition', 'Sensor'.
        """
        values = 'NormalValue' if normalize else 'RawValue'
        sensor_type = 'Region' if agg_regions else 'Electrode'
        
        if agg_regions:
            FEUtilz.get_regions(feature_table)
            
        if get_diffs:
            conditions_split = feature_table.pivot_table(index=['Subject', sensor_type, 'FeatureName']
                    , columns='Condition'
                    , values='RawValue'
                    ).reset_index()
            conditions_split['AbsoluteDIfference'] = np.abs(conditions_split['walk'] - conditions_split['sit'])

            pivoted = conditions_split.pivot_table(
                index=['Subject']
                , columns=['FeatureName', sensor_type]
                , values='AbsoluteDIfference'
            )
        else:
            pivoted = feature_table.pivot_table(
                index='Subject'
                , columns=['FeatureName', 'Condition', sensor_type]
                , values=values
                )
        
        feature_mat = pivoted.values 
        subjects = pivoted.index.values
        feature_names = np.array(list(pivoted.columns))
        
        return feature_mat, subjects, feature_names
    
    @staticmethod
    def is_healthy(feature_table: pd.DataFrame):
        """
        Create a boolean column for subject type - a healthy control (HC; True) of parkinsons patient (PD; False).
        :param feature_table: a dataframe representing the features.
        """
        feature_table['IsHC'] = feature_table['Subject'].str.startswith('HC')

    @staticmethod
    def store_params_per_category(obj, category: str, params: dict):
        """
        A helper function, gets an analyzer object and saves all its attributes into a dictionary.
        :param obj: the analyzer object.
        :param category: a string representing the category name of the analyzer.
        :param params: a dictionary to store attributes into.
        """
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            if not attr_name.startswith('__') and not callable(attr):
                params[category].update({attr_name: attr})
    
    @staticmethod
    def get_regions(feature_table: pd.DataFrame):
        """
        Creates a new column of the brain regions corresponding to an electrodes column.
        :param feature_table: a dataframe representing the features.
        """
        feature_table['Region'] = feature_table['Electrode'].apply(lambda elec: FEUtilz.ELECTRODE_REGION_MAP.get(elec, 'Unknown'))


class ERPAnalysis:
    def __init__(self, sampling_rate: int, time_windows: dict[str: tuple[int]], electrodes: dict[str, list[str]], components_signs: dict[str: bool]
                 , time_offset: int, min_peak_width: int, min_peak_height: float, smoothness_span: int, area_fractions: list[float]
                 , conditions: list[str], difference_electrodes: dict[str: tuple[str]] = None, difference_conditions: list[tuple[str]] = None):
        """
        Initializing an ERPAnalysis object with necessary parameters for computations.Notice that a sampling rate of 250Hz is assumed.
        :param sampling_rate: the sampling rate of the signal in Hz.
        :param time_windows: a dictionary {ERP_component_name: (start_time, end_time)} describing the time windows for all desired ERP components.
        notice times should be given in milliseconds.
        :param electrodes: a dictionary {ERP_components_name: electrodes_names} listing the relevant electrodes for all desired components.
        :param components_signs: a dictionary {ERP_components_name: ERP_components_sign} describing the sign (+ \ -) of all desired ERP components.
        Notice that ERP_components_sign is a boolean where True = +, and False = -.
        :param time_offset: the duration of the baseline recordings in milliseconds.
        :param min_peak_width: an integer representing the minimal width of a peak in an ERP component.
        :param min_peak_height: a float representing the minimal height of a peak in an ERP component.
        :param smoothness_span: an integer representing the size of the time-series environment used for smoothing the peak amplitude
        , around the detected peak. The returned amplitude will be an average of values within this environment. 
        The environment is bi-directional. Default value is 3.
        :param area_fractions: a list of desired fractions to be calculated in the fractional area analysis.
        :param conditions: list of conditions names in the experiment.
        :param difference_electrodes: a dictionary {ERP_components_name: (first_electrode, second_electrode)} for lateralization 
        effect difference wave calculation. Default is None.
        :param difference_conditions: a list of tuples [(cond1, cond2)] representing the desired conditions to calculate a difference wave between.
        The difference wave will be calculated as (cond1 - cond2). Default is None.
        """
        self.sampling_rate = sampling_rate
        self.temporal_resolution = 1000 / self.sampling_rate # [ms]
        self.components = list(time_windows.keys())
        self.time_windows = time_windows
        self.electrodes = electrodes 
        self.components_signs = components_signs
        self.time_offset = time_offset
        self.min_peak_width = min_peak_width
        self.min_peak_height = min_peak_height
        self.smoothness_span = smoothness_span
        self.area_fractions = np.array(area_fractions)
        self.conditions = conditions
        self.difference_electrodes = difference_electrodes
        self.difference_conditions = difference_conditions

    def get_peak_features(self, epochs_data: np.ndarray, electrodes: list[str], is_max: bool
                          , feature_category: str, feature_name: str, time_offset: int, component: str) -> pd.DataFrame:
        """
        Calculate the peak amplitude and peak latency within the given waveform
        , by finding the most extreme peak in the desired direction within each epoch
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the averaged EEG waveform across epochs
        , sliced to the desired time-window of the ERP component.
        :param electrodes: a list of the names of electrodes corresponding to the sub-arrays in epochs_data.
        :param is_max: true if searching for a maximum peak, false if searching for a minimum peak.
        :param min_width: an integer representing the minimal width of a peak.
        :param min_height: a float representing the minimal height of a peak.
        :param feature_category: the name of the feature category for the feature table.
        :param feature_name: the name of the feature itself for the feature table.
        :param time_offset: an integer representing the time offset in milliseconds of the epochs data.
        :param component: a string representing the ERP component for the analysis.
        :return: a dataframe with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single peak measure (amplitude or latency).
        """
        peak_features = np.apply_along_axis(self.calc_single_peak, axis=1, arr=epochs_data, is_max=is_max, time_offset=time_offset
                                            , component=component)
        amplitude_table = FEUtilz.build_feature_df(electrodes, feature_name + '_absolute_peak_amplitude'
                                                   , feature_category, peak_features[:, 0])
        latency_table = FEUtilz.build_feature_df(electrodes, feature_name + '_peak_latency'
                                                 , feature_category, peak_features[:, 1])

        return pd.concat([amplitude_table, latency_table]).reset_index(drop=True)

    def calc_single_peak(self, epoch_data: np.ndarray, is_max: bool, time_offset: int, component: str) -> np.ndarray:
        """
        Calculating the peak measures of a single epoch, by finding the most extreme peak in the desired direction.
        :param epoch_data: a 1-dimensional array of the averaged EEG waveform across epochs from a single electrode.
        :param is_max: true if searching for a maximum peak, false if searching for a minimum peak.
        :param time_offset: an integer representing the time offset in milliseconds of the epochs data.
        :param component: a string representing the ERP component for the analysis.
        :return: a (2,) array with the peak amplitude and peak latency.
        If no peak is found, amplitude is set to 0 and latency is set to the component mid-point ((time_start + time_end) / 2).
        """
        time_window = self.time_windows[component]
        mid_point = (time_window[1] + time_window[0]) / 2
        
        # negate the waveform for min objective
        data = epoch_data if is_max else epoch_data * (-1)   

        peak_indices, metadata = scipy.signal.find_peaks(data, width=self.min_peak_width, height=self.min_peak_height)
        
        # getting actual peak measures
        if peak_indices.size > 0: # some peaks were found 
            argopt = metadata['peak_heights'].argmax()        
            
            # if found only non-relevant peaks, discard them
            if metadata['peak_heights'][argopt] < 0:
                return np.nan(2)
            
            sample_latency = peak_indices[argopt]
            time_latency = sample_latency * self.temporal_resolution + time_offset
            
            peak_environment = [sample_latency + i for i in range(- self.smoothness_span, self.smoothness_span + 1) ]
            peak_environment = [i for i in peak_environment if 0 <= i and i < data.shape[0]]
            amplitude = data[peak_environment].mean()
        
        else: # no peaks were found
            return np.full(2, np.nan)
            
        return np.array([amplitude, time_latency])

    def get_area_features(self, epochs_data: np.ndarray, electrodes: list[str], feature_category: str
                          , feature_name: str, time_offset: int, is_max: bool = True, is_diff_wave: bool = False) -> pd.DataFrame:
        """
        Calculate the overall amplitude by signed AUC and fractional latency using percentiles.
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the EEG waveform
        , sliced to the desired time-window of the ERP component.
        :param electrodes: a list of the names of electrodes corresponding to the sub-arrays in epochs_data.
        :param feature_category: the name of the feature category for the feature table.
        :param feature_name: the name of the feature itself for the feature table.
        :param time_offset: an integer representing the time offset in milliseconds of the epochs data.
        :param is_max: true if searching for a maximum peak, false if searching for a minimum peak.
        Default is True.
        :param is_diff_wave: true for analyzing difference wave. if true, disregards 'is_max' and only 
        considers the absolute value waveform. Default is False.
        :return: a dataframe with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single area measure (amplitude or latency with a certain percentile).
        """
        area_features = np.apply_along_axis(self.calc_single_area, axis=1, arr=epochs_data, time_offset=time_offset
                                            , is_max=is_max, is_diff_wave=is_diff_wave)
        feature_table = FEUtilz.build_feature_df(electrodes, feature_name + '_mean_signed_AUC'
                                                    , feature_category, area_features[:, 0])
        
        all_feature_tables = [feature_table]
        for i in range(len(self.area_fractions)):
            latency_table = FEUtilz.build_feature_df(electrodes
                                                    , feature_name + f'_fractional_area_latency_{self.area_fractions[i]}'
                                                    , feature_category, area_features[:, i + 1])
            all_feature_tables.append(latency_table)
            
        feature_table = pd.concat(all_feature_tables)
        return feature_table

    def calc_single_area(self, epoch_data: np.ndarray, time_offset: int, is_max: bool = True, is_diff_wave: bool = False) -> np.ndarray:
        """
        Calculate the overall amplitude by mean signed AUC and fractional latency using percentiles.
        :param epochs_data: a 1-dimensional array (samples) of the EEG waveform
        , sliced to the desired time-window of the ERP component.
        :param time_offset: an integer representing the time offset in milliseconds of the epochs data.
        :param is_max: true if searching for a maximum peak, false if searching for a minimum peak.
        Default is True.
        :param is_diff_wave: true for analyzing difference wave. if true, disregards 'is_max' and only 
        considers the absolute value waveform. Default is False.
        :return: a (1 + k,) array with the mean signed AUC and fractional area latencies (k).
        where each row corresponds to a single area measure (amplitude or latency with a certain percentile).
        """
        if is_diff_wave:
            # consider only the absolute value of the waveform, for absolute area
            data = abs(epoch_data)
        else:
            # get only direction-relevant amplitudes, negate data for negative components
            data = np.maximum(epoch_data, 0) if is_max else np.minimum(epoch_data, 0) * (-1)  
        
        cum_integral = scipy.integrate.cumulative_trapezoid(data, initial=0)
        signed_AUC = cum_integral[-1]
        mean_signed_AUC = signed_AUC / cum_integral.shape[0]
        
        # calculating percentiles and converting them into times
        fractional_area_latencies = np.argmin(cum_integral[:, np.newaxis] < signed_AUC * self.area_fractions, axis=0)
        fractional_area_latencies = fractional_area_latencies * self.temporal_resolution + time_offset
        
        return np.concatenate((np.array([mean_signed_AUC]), fractional_area_latencies))
        
    @staticmethod
    def calc_electrodes_difference_wave(epochs_data: np.ndarray, first_electrode: int, second_electrode: int
                                        , electrodes_map: list) -> np.ndarray:
        """
        Calculating a difference wave between 2 given electrodes (first - second) that are found within the EEG data.
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the EEG waveform.
        :param first_electrode: the name of the first electrode (e.g. 'Cz').
        :param second_electrode: the name of the second electrode (e.g. 'Cz').
        :param electrodes_map: the list of all electrodes included in the EEG data tensor.
        :return: a 2-dimensional array of (1, samples) of the difference wave between (first_electrode - second_electrode).
        In addition, returns a valid boolean which is True if and only if the electrodes exist in the data.
        """
        first_index, _ = FEUtilz.electrode_name_translate([first_electrode], electrodes_map)
        second_index, _ = FEUtilz.electrode_name_translate([second_electrode], electrodes_map)
        valid = (len(first_index) == 1) and (len(second_index) == 1)
        return epochs_data[first_index] - epochs_data[second_index], valid
    
    def get_cond_diff_wave_features(self, subject_data_cache: dict[str: np.ndarray], electrodes_map: dict[str: list[str]]):
        """
        Calculating difference wave features between all given experimental conditions, within a given subject.
        :param subject_data_cache: a dictionary {cond_comp: epoch_data} where cond_comp is a joint string representation of the
        experimental condition and the relevant ERP component. epochs_data should be a 2-dimensional array with (electrodes, samples).
        :param electrodes_map: a dictionary {cond_comp: elec_map} where cond_comp is a joint string representation of the
        experimental condition and the relevant ERP component. elec_map should be a list of the names of electrodes corresponding 
        to the relevant epoch_data.
        :return: a dataframe with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        """
        all_diff_tables = []
        
        for cond1, cond2 in self.difference_conditions:
            
            for comp in self.components:                
                key1 = f'{cond1}_{comp}'
                key2 = f'{cond2}_{comp}'
                
                if key1 in electrodes_map and key2 in electrodes_map:
                    # getting averaged and sliced data from cache
                    elec_map1 = electrodes_map[key1]
                    elec_map2 = electrodes_map[key2]
                    joint_elec_map = elec_map1
                else:
                    continue
                
                if  key1 in subject_data_cache and key2 in subject_data_cache:
                    cond1_epochs_data = subject_data_cache[key1]
                    cond2_epochs_data = subject_data_cache[key2]
                else:
                    continue

                
                # making sure that both conditions have the same shape
                if len(elec_map1) < len(elec_map2):
                    cond2_epochs_data, joint_elec_map = FEUtilz.slice_epochs_2d(cond2_epochs_data, sampling_rate=self.sampling_rate, start_time=np.inf
                                                                                , end_time=np.inf, time_offset=self.time_offset
                                                                                , electrodes_map=elec_map2, electrode_names=elec_map1)
                elif len(elec_map1) > len(elec_map2):
                    cond1_epochs_data, joint_elec_map = FEUtilz.slice_epochs_2d(cond1_epochs_data, sampling_rate=self.sampling_rate, start_time=np.inf
                                                                                , end_time=np.inf, time_offset=self.time_offset
                                                                                , electrodes_map=elec_map1, electrode_names=elec_map2)
                
                # calculating difference wave feature
                comp_time_offset = self.time_windows[comp][0]
                
                cond_diff_wave = cond1_epochs_data - cond2_epochs_data
                curr_feature_table = self.get_area_features(cond_diff_wave, electrodes=joint_elec_map, is_diff_wave=True, time_offset=comp_time_offset
                                                , feature_category=f'ERP', feature_name='conditions_difference_wave')
                
                curr_feature_table['Component'] = comp
                curr_feature_table['Condition'] = f'{cond1} - {cond2}'
                all_diff_tables.append(curr_feature_table)
        
        return pd.concat(all_diff_tables) if len(all_diff_tables) > 0 else pd.DataFrame()
    
    def get_ERP_features(self, epochs_data: np.ndarray, electrodes_map: list[str], component: str) -> pd.DataFrame:
        """
        Calculating all ERP features from a single subject, condition and ERP component related electrodes, into a dataframe.
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the averaged EEG waveform, within the relevant electrodes and time window.
        :param electrodes_map: a list of the names of electrodes corresponding to the sub-arrays in epochs_data.
        :param component: the name of ERP component according to which to calculate features.
        :return: a dataframe with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        """
        component_time_offset = self.time_windows[component][0]
        
        # calculating peak features
        is_max = self.components_signs[component]
        peak_table = self.get_peak_features(epochs_data, electrodes_map, is_max=is_max, time_offset=component_time_offset
                                            , feature_category=f'ERP', feature_name=component, component=component)
        
        # calculating area features
        area_table = self.get_area_features(epochs_data, electrodes_map, is_max=is_max, time_offset=component_time_offset
                                            , feature_category=f'ERP', feature_name=component)
        
        # calculating electrode difference wave features
        first_electrode, second_electrode = self.difference_electrodes[component]
        electrodes_diff_wave, valid = ERPAnalysis.calc_electrodes_difference_wave(epochs_data, first_electrode
                                                                    , second_electrode, electrodes_map=electrodes_map)
        if valid:
            diff_feature_table = self.get_area_features(electrodes_diff_wave, electrodes=[f'{first_electrode} - {second_electrode}']
                                    , time_offset=component_time_offset, is_diff_wave=True, feature_category=f'ERP', feature_name=f'{component}_electrodes_difference_wave')
        else:
            diff_feature_table = pd.DataFrame()
            
        ERP_feature_table = pd.concat([peak_table, area_table, diff_feature_table])
        return ERP_feature_table


class ComplexityAnalysis:
    def __init__(self, kmax: int, embedding_dim: int, tau: int, normalize: bool = True, calc_std: bool = True, electrodes: list[str] = []):
        """
        Initializing an ComplexityAnalysis object with necessary parameters for computations. Notice that a sampling rate of 250Hz is assumed.
        :param kmax: an integer representing kmax hyperparameter for Higuchi's Fractal Dimension calculation.
        :param embedding_dim: an integer representing the embedding dimension used for Permutation Entropy calculation.
        :param tau: an integer representing the time scale in samples for Permutation Entropy calculation.
        :param normalize: if True, normalizes Lempel-Ziv Complexity values, Default is True.
        :param calc_std: specify whether to calculate the standard deviation (STD) of complexity features, or otherwise only the mean.
        Default is True.
        :param electrodes: the electrode names from which to calculate temporal complexity features.
        """
        # General Parameteres
        self.electrodes = electrodes
        self.calc_std = calc_std
        
        # Higuchi's Fractal Dimension Parameters
        self.kmax = kmax
        
        # Permutation Entropy Parameters 
        self.embedding_dim = embedding_dim
        self.tau = tau
        
        # Lempel-Zic-Complexity Parameters
        self.normalize = normalize
    
    def _calc_HFD(self, epoch):
        return higuchi_fd(epoch, kmax=self.kmax)
    
    def _calc_perm_entropy(self, epoch):
        return PermEn(epoch, m=self.embedding_dim, tau=self.tau)[1][-1]
    
    def _calc_LZC(self, epoch):
        return lziv_complexity(epoch > epoch.mean(), normalize=self.normalize)
    
    def _calc_complexity_measure(self, complexity_func, epochs_data: np.ndarray, electrodes: list[str], feature_name: str) -> list[pd.DataFrame]:
        """ 
        Calculate the mean and standard deviation of a given temporal complexity function of the EEG waveform.
        :param complexity_func: a callable representing a function the takes only an epoch and computes a complexity measure.
        :param epochs_data: a 3-dimensional array (trials, electrodes, samples) of the EEG waveform.
        :param electrodes: a list of the names of electrodes corresponding to the sub-arrays in epochs_data.
        :param feature_name: the name of the feature itself for the feature table.
        :return: a list of dataframes with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single statistical measure (mean, standard deviation) of the temporal complexity. 
        within a specific electrode, across epochs.
        """
        complexities = np.apply_along_axis(complexity_func, arr=epochs_data, axis=2)
        
        mean_table = FEUtilz.build_feature_df(electrodes, feature_name=f'{feature_name}_mean'
                                 , feature_category='TemporalComplexity', values=complexities.mean(axis=0))
        if self.calc_std:
            std_table = FEUtilz.build_feature_df(electrodes, feature_name=f'{feature_name}_std'
                                    , feature_category='TemporalComplexity', values=complexities.std(axis=0))
            return [mean_table, std_table]
        else:
            return [mean_table]    
    
    def get_complexity_features(self, epochs_data: np.ndarray, electrodes_map: list[str]) -> pd.DataFrame:
        """
        Calculating Temporal Complexity features from a single subject, condition and ERP component related electrodes, into a dataframe.
        :param epochs_data: a 3-dimensional array (trials, electrodes, samples) of the EEG waveform, within the relevant electrodes and time window.
        :param electrodes_map: a list of the names of electrodes corresponding to the sub-arrays in epochs_data.
        :return: a dataframe with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will be 'TemporalComplexity'.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        """
        # calculating temporal complexity features
        HFD_tables = self._calc_complexity_measure(self._calc_HFD, epochs_data, electrodes_map, feature_name=f'HFD')
        LZC_tables = self._calc_complexity_measure(self._calc_LZC, epochs_data, electrodes_map, feature_name=f'LZC')
        PermEn_tables = self._calc_complexity_measure(self._calc_perm_entropy, epochs_data, electrodes_map, feature_name=f'PermEn')
        
        # concatenating feature tables
        feature_table = pd.concat(HFD_tables + LZC_tables + PermEn_tables, ignore_index=True)
        
        return feature_table


class FrequencyAnalysis:
    def __init__(self, sampling_rate: int, samples_per_segment: int, frequency_range: tuple[float], frequency_bands: dict[str: list[float]]
                 , peak_width_limits: list = (1, 12), max_n_peaks: int = np.inf, min_peak_height: float = 0, min_r_squared: float = 0
                 , aperiodic_mode: str = 'fixed', electrodes: list[str] = []):
        """
        Initializing an FrequencyAnalysis object with necessary parameters for computations.
        :param sampling_rate: the sampling rate of the signal in Hz.
        :param samples_per_segment: number of samples per segment used for Power Spectral Density (PSD) computation.
        :param frequency_range: a tuple of floats (lower, upper) representing the lowe and upper bound for frequencies to include in analysis.
        :param frequency_bands: a dictionary {band_name: [lower, upper]} representing frequency band names and corresponding lower and upper 
        frequency boundaries.
        :param peak_width_limits: the minimal and maximal peak widths for FOOOF.
        :param max_n_peaks: maximal number of peaks to fit in the FOOOF computation, on each PSD.
        :param min_peak_height: a float representing the minimal height for a peak in the FOOOF computation.
        :param min_r_squared: the minimal accuracy score (r_squared) for FOOOF parametrization. If actual accuracy is lower, discards the specific
        calculated parametrization. Should be between 0 and 1.
        :param aperiodic_mode: the mode of the aperiodic fit {'knee', 'fixed'}.
        :param electrodes: the electrode names from which to calculate frequency features.
        """
        # General parameters
        self.electrodes = electrodes
        
        # PSD parameters
        self.sampling_rate = sampling_rate
        self.samples_per_segment = samples_per_segment
        
        # FOOOF parameters
        self.frequency_range = frequency_range
        self.peak_width_limits = peak_width_limits
        self.max_n_peaks = max_n_peaks
        self.min_peak_height = min_peak_height
        self.frequency_bands = Bands(frequency_bands)
        self.aperiodic_mode = aperiodic_mode
        self.min_r_squared = min_r_squared
        
        self.fooof_accuracy = pd.DataFrame()
        
    def reset_accuracy(self):
        """
        Resets the fooof_accuracy attribute.
        """
        self.fooof_accuracy = pd.DataFrame({'error': [], 'r_squared': []})
    
    def get_PSD(self, epochs_data: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculating Power Spectral Density (PSD) from the EEG data. 
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the EEG waveform.
        :return freqs: a 1-dimensional array of sampled frequencies used for PSD calculation.
        :return powers: a 2-dimensional array of (electrodes, powers) where each power corresponds
        to a frequency on freqs.
        """       
        freqs, powers = scipy.signal.welch(epochs_data, fs=self.sampling_rate, nperseg=self.samples_per_segment)
        lower, upper = self.frequency_range
        freq_filter = (freqs > lower) & (freqs < upper)
        
        return freqs[freq_filter], powers[:, freq_filter]
    
    def get_FOOOF_features(self, freqs: np.ndarray, powers: np.ndarray, electrodes: list[str]
                           , feature_name: str) -> list[pd.DataFrame]:
        """
        Train a FOOOF model on the EEG data to get periodic and aperiodic features in the frequency-domain.
        :param freqs: a 1-dimensional array of a linear space of frequencies, on which PSD (Power Spectral Density) 
        calculation was performed.
        :param powers: a 2-dimensional array of (electrodes, powers) where each power corresponds to a specific frequency
        within a specific electrode.
        :param electrodes: a list of the names of electrodes corresponding to the sub-arrays in powers.
        :param feature_name: the name of the feature itself for the feature table.
        :return: a list of dataframes with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will be 'FrequencyDomain'.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        """
        # training FOOOF model for each electrode
        spec_model = SpectralGroupModel(peak_width_limits=self.peak_width_limits, min_peak_height=self.min_peak_height
                                   , max_n_peaks=self.max_n_peaks, aperiodic_mode=self.aperiodic_mode, verbose=False)
        spec_model.fit(freqs, powers, freq_range=list(self.frequency_range))
        
        # parsing the feature table
        fooof_params = spec_model.to_df(self.frequency_bands)
        
        accuracy_mask = fooof_params['r_squared'] >= self.min_r_squared
        fooof_params = fooof_params.loc[accuracy_mask]
        
        accuracy_params = ['error', 'r_squared']
        self.fooof_accuracy = pd.concat([self.fooof_accuracy, fooof_params[accuracy_params]])
        
        param_cols = [col for col in fooof_params.columns if col not in accuracy_params]
        all_fooof_tables = []
        for col in param_cols:
            col_feature_table = FEUtilz.build_feature_df(electrodes, feature_category='FrequencyDomain', 
                                     feature_name=f'{feature_name}_{col}', values=fooof_params[col].values)
            all_fooof_tables.append(col_feature_table)
        
        return all_fooof_tables
    
    def get_powerbands(self, freqs: np.ndarray, powers: np.ndarray, electrodes: list[str]
                           , feature_name: str) -> list[pd.DataFrame]:
        """
        Calculates classical relative powerbands on a given Power Spectral Density (PSD) (e.g. beta relative power).
        :param freqs: a 1-dimensional array of a linear space of frequencies, on which PSD (Power Spectral Density) 
        calculation was performed.
        :param powers: a 2-dimensional array of (electrodes, powers) where each power corresponds to a specific frequency
        within a specific electrode.
        :param electrodes: a list of the names of electrodes corresponding to the sub-arrays in powers.
        :param feature_name: the name of the feature itself for the feature table.
        :return: a list of dataframes with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will be 'FrequencyDomain'.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        """
        all_powerband_tables = []
        
        total_powers = np.trapz(powers, freqs, axis=1)
        total_table = FEUtilz.build_feature_df(electrodes, feature_category='FrequencyDomain', 
                                    feature_name=f'total_{feature_name}', values=total_powers)
        all_powerband_tables.append(total_table)
        
        for band_name, (lower, upper) in self.frequency_bands.bands.items():
            # Get relative powerband
            band_mask = (lower <= freqs) & (freqs <= upper)
            absolute_bandpowers = np.trapz(powers[:, band_mask], freqs[band_mask], axis=1)
            relative_bandpowers = absolute_bandpowers / total_powers
            
            # Buiild features dataframe
            band_table = FEUtilz.build_feature_df(electrodes, feature_category='FrequencyDomain', 
                                     feature_name=f'{band_name}_relative_{feature_name}', values=relative_bandpowers)
            all_powerband_tables.append(band_table)
        
        return all_powerband_tables
    
    def get_frequency_features(self, epochs_data: np.ndarray, electrodes_map: list[str]) -> pd.DataFrame:
        """
        Calculating all frequency-domain features from a single subject, condition and ERP component related electrodes, into a dataframe.
        :param epochs_data: a 2-dimensional array (electrodes, samples) of the averaged EEG waveform, within the relevant electrodes and time window.
        :param electrodes_map: a list of the names of electrodes corresponding to the sub-arrays in epochs_data.
        :return: a dataframe with the following fields - Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        """
        # performing PSD calculation
        freqs, powers = self.get_PSD(epochs_data)
        
        # getting fooof feature table
        fooof_tables = self.get_FOOOF_features(freqs=freqs, powers=powers, electrodes=electrodes_map
                                                , feature_name=f'fooof')
        
        powerband_tables = self.get_powerbands(freqs=freqs, powers=powers, electrodes=electrodes_map
                                                , feature_name=f'bandpower')
        
        return pd.concat(fooof_tables + powerband_tables)


class FeatureExtractor:
    def __init__(self, eeglab_io: EEGLABIO, results_dir: str = RES_DIR):
        """
        Abstract class for FeatureExtractor objects.
        :param eeglab_io: a EEGLABIO object to handle I\O operations.
        :parma results_dir: a path to the results directory, in which to store the table.
        """
        self.eeglab_io = eeglab_io
        self.file_paths = self.eeglab_io.get_file_paths_by_subject()
        self.conditions = self.eeglab_io.conditions
        
        self.results_dir = results_dir
        
        self.subjects_metadata = {}
        self.reset_subjects_metadata()
    
    def update_subjects_metadata(self, subject: str, subject_data: dict[str: str]):
        """
        Updating the subjects metadata attribute for a single subject's data.
        :param subject: a string representation of the subject's ID, as extracted from the file name.
        :param subject_data: a dictionary {condition: file_path} representing the raw data file paths of the given subject
        per condition.
        """
        for cond, data in subject_data.items():
            self.subjects_metadata['Subject'].append(subject)
            self.subjects_metadata['Condition'].append(cond)
            self.subjects_metadata['NElectrodes'].append(data.shape[1])
            self.subjects_metadata['NTrials'].append(data.shape[0])
    
    def reset_subjects_metadata(self):
        """
        Resets the subjects_metadata attribute.
        """
        self.subjects_metadata = {
            'Subject': []
            , 'Condition': []
            , 'NElectrodes': []
            , 'NTrials': []
        }


class EpochsFeatureExtractor(FeatureExtractor):
    def __init__(self, ERP_analyzer: ERPAnalysis, complexity_analyzer: ComplexityAnalysis, frequency_analyzer: FrequencyAnalysis
                 , eeglab_io: EEGLABIO, time_window: (float, float), results_dir: str = RES_DIR):
        """
        Initializing a FeatureExtractor for the analysis of Evoked Potentials, organized into epochs.
        :param ERP_analyzer: an ERPAnalysis object, used to calculate ERP features.
        :param complexity_analyzer: an ComplexityAnalysis object, used to calculate temporal complexity features.
        :param frequency_analyzer: an FrequencyAnalysis object, used to calculate FOOOF features.
        :param eeglab_io: an EEGLABIO object for I\O handling.
        :param time_window: a tuple (start_time, end_time) representing the time-window for computation in milliseconds.        
        :parma results_dir: a path to the results directory, in which to store the table.
        """
        super().__init__(eeglab_io, results_dir)
        self.ERP_analyzer = ERP_analyzer
        self.complexity_analyzer = complexity_analyzer
        self.frequency_analyzer = frequency_analyzer
        
        assert self.complexity_analyzer.electrodes == self.frequency_analyzer.electrodes
        self.electrodes = self.complexity_analyzer.electrodes

        self.time_window = time_window  # Without baseline
        self.time_offset = self.ERP_analyzer.time_offset
        
        assert self.ERP_analyzer.sampling_rate == self.frequency_analyzer.sampling_rate
        self.sampling_rate = self.ERP_analyzer.sampling_rate
        
        self.store_params()
        
    def get_subject_feaures(self, subject: str, calc_ERP: bool, calc_complexity: bool, calc_frequency: bool):
        """
        Getting a full feature table from a single subject from all categories.
        :param subject: a string representing the subjectID.
        :param calc_ERP: specify if classic ERP features should be calculated.
        :param calc_complexity: specify if temporal complexity features should be calculated.
        :param calc_frequency: specify if fooof-based frequency features should be calculated.
        :return: a list of dataframes, each with the following fields - Subject, Condition, Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Subject is the subject's ID as was extracted from the raw data file name.
        -- Condition is the experimental condition name.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        -- NormalValue is the raw feature value after a softmax transformation.
        """
        suject_feature_tables = []
        subject_data_cache = {}
        electrode_map_cache = {}
        
        try:
            subject_data, electrodes_map = EEGLABIO.get_subject_data(self.file_paths[subject], is_epochs=True)
            self.update_subjects_metadata(subject, subject_data)
            
            for cond in self.conditions:   
                if cond not in subject_data:
                    continue
                
                else:
                    cond_data = subject_data[cond]
                    cond_elec_map = electrodes_map[cond]
                    FEUtilz.convert_electrode_names(cond_elec_map)    
                
                start_time, end_time = self.time_window

                # slicing data for complexity and frequnecy features
                if calc_complexity or calc_frequency:
                    sliced_data, updated_elec_map = FEUtilz.slice_epochs_3d(epochs_data=cond_data, sampling_rate=self.sampling_rate
                                                                            , start_time=start_time, end_time=end_time
                                                                            , time_offset=self.time_offset, electrodes_map=cond_elec_map
                                                                            , electrode_names=self.electrodes)
                
                # calculating Temporal Complexity features
                if calc_complexity:
                    complexity_table = self.complexity_analyzer.get_complexity_features(sliced_data, updated_elec_map)
                else:
                    complexity_table = pd.DataFrame()
                    
                # calculating Frequency features
                if calc_frequency:
                    averaged_for_frequency = FEUtilz.average_epochs(sliced_data)
                    frequency_table = self.frequency_analyzer.get_frequency_features(averaged_for_frequency, updated_elec_map)
                else:
                    frequency_table = pd.DataFrame()
                
                if calc_ERP:                    
                    ERP_tables = []
                    for comp in self.ERP_analyzer.components:
                        # slicing only for component-relevant electrodes and times
                        start_time, end_time = self.ERP_analyzer.time_windows[comp]
                        sliced_for_ERP, updated_elec_map = FEUtilz.slice_epochs_3d(epochs_data=cond_data, sampling_rate=self.sampling_rate
                                                                                    , start_time=start_time, end_time=end_time
                                                                                    , time_offset=self.time_offset, electrodes_map=cond_elec_map
                                                                                    , electrode_names=self.ERP_analyzer.electrodes[comp])
                        averaged_for_ERP = FEUtilz.average_epochs(sliced_for_ERP)
                        
                        # caching the data and electrode map for later use
                        subject_data_cache.update({f'{cond}_{comp}': averaged_for_ERP})
                        electrode_map_cache.update({f'{cond}_{comp}': updated_elec_map})
                        
                        # calculating ERP features
                        ERP_table = self.ERP_analyzer.get_ERP_features(averaged_for_ERP, updated_elec_map, comp)
                        ERP_table['Component'] = comp
                        ERP_tables.append(ERP_table)
                else:
                    ERP_tables = []

                # concatenating data frames
                curr_feature_table = pd.concat([complexity_table, frequency_table] + ERP_tables)
                curr_feature_table['Condition'] = cond
                curr_feature_table['Subject'] = subject
                    
                # Caching dataframe
                suject_feature_tables.append(curr_feature_table)
        
            # calculating ERP condition difference wave features, separate for efficiency
            if calc_ERP and len(subject_data_cache) > 0:
                diff_wave_table = self.ERP_analyzer.get_cond_diff_wave_features(subject_data_cache, electrode_map_cache)
            else:
                diff_wave_table = pd.DataFrame()
            
            # Caching dataframe
            diff_wave_table['Subject'] = subject
            suject_feature_tables.append(diff_wave_table)
        
        except Exception as e:
            warnings.warn(f"***Problem with {subject}, continued to the next.")
        
        return suject_feature_tables
    
    def get_feature_table(self, calc_ERP: bool = True, calc_complexity: bool = True, calc_frequency: bool = True) -> pd.DataFrame:
        """
        Getting a full feature tables from all categories.
        :param calc_ERP: specify if classic ERP features should be calculated.
        :param calc_complexity: specify if temporal complexity features should be calculated.
        :param calc_frequency: specify if fooof-based frequency features should be calculated.
        :return: a dataframe with the following fields - Subject, Condition, Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Subject is the subject's ID as was extracted from the raw data file name.
        -- Condition is the experimental condition name.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        -- NormalValue is the raw feature value after a softmax transformation.
        """
        all_feature_tables = []
        self.reset_subjects_metadata()
        self.frequency_analyzer.reset_accuracy()
        
        subjects = self.file_paths.keys()
        
        all_feature_tables = []
        for subject in tqdm(subjects, desc='Computing features per subject'):
            subject_feature_table = self.get_subject_feaures(subject, calc_ERP, calc_complexity, calc_frequency)
            all_feature_tables.extend(subject_feature_table)
        
        # concatenating data frames
        feature_table = pd.concat(all_feature_tables)
        feature_table.drop_duplicates(ignore_index=True, inplace=True)
        FEUtilz.normalize(feature_table)
        FEUtilz.is_healthy(feature_table)
        
        FEUtilz.store_table(feature_table, 
                            file_name='epochs_feature_table', 
                            results_dir=self.results_dir
                            )
        
        FEUtilz.store_table(pd.DataFrame(self.subjects_metadata), 
                            file_name='subjects_metadata_epochs', 
                            results_dir=self.results_dir
                            )
        
        FEUtilz.store_table(self.frequency_analyzer.fooof_accuracy.reset_index(drop=True), 
                            file_name='fooof_accuracy_epochs',
                            results_dir=self.results_dir
                            )
        
        return feature_table
    
    def store_params(self):
        """
        Stores all the parameters used to calculate features into a single .txt file.
        """
        params = {
            'ERP': {}
            , 'Complexity': {}
            , 'Frequency': {}
        }
        
        FEUtilz.store_params_per_category(self.ERP_analyzer, 'ERP', params)
        FEUtilz.store_params_per_category(self.complexity_analyzer, 'Complexity', params)
        FEUtilz.store_params_per_category(self.frequency_analyzer, 'Frequency', params)
        
        timestamp = dt.datetime.now().strftime('%d_%m_%Y_%H_%M')
        path = os.path.join(self.results_dir, rf"epochs_model_parameters_{timestamp}.txt")
        with open(path, "w") as file:
            for category, data in params.items():
                file.write(f"----{category}:\n")
                for attr_name, attr in data.items():
                    file.write(f"{attr_name}: {attr}\n")
                file.write(f"\n")
        
        print(f'saved file at model_parameters_{timestamp}.txt')


class RestingStateFeatureExtractor(FeatureExtractor):
    def __init__(self, complexity_analyzer: ComplexityAnalysis, frequency_analyzer: FrequencyAnalysis, eeglab_io: EEGLABIO
                 , results_dir: str = RES_DIR):
        """
        Initializing a FeatureExtractor for the analysis of Resting State potentials.
        :param complexity_analyzer: an ComplexityAnalysis object, used to calculate temporal complexity features.
        :param frequency_analyzer: an FrequencyAnalysis object, used to calculate FOOOF features.
        :param eeglab_io: an EEGLABIO object for I\O handling.
        :parma results_dir: a path to the results directory, in which to store the table.
        """
        super().__init__(eeglab_io, results_dir)
        self.complexity_analyzer = complexity_analyzer
        self.frequency_analyzer = frequency_analyzer
        
        assert self.complexity_analyzer.electrodes == self.frequency_analyzer.electrodes
        self.electrodes = self.complexity_analyzer.electrodes
        
        self.sampling_rate = self.frequency_analyzer.sampling_rate
        
        self.store_params()
    
    def get_subject_feaures(self, subject: str, calc_complexity: bool, calc_frequency: bool):
        """
        Getting a full feature table from a single subject from all categories.
        :param subject: a string representing the subjectID.
        :param calc_complexity: specify if temporal complexity features should be calculated. Should satisfy calc_std = False.
        :param calc_frequency: specify if fooof-based frequency features should be calculated.
        :return: a list of dataframes, each with the following fields - Subject, Condition, Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Subject is the subject's ID as was extracted from the raw data file name.
        -- Condition is the experimental condition name.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        -- NormalValue is the raw feature value after a softmax transformation.
        """        
        suject_feature_tables = []
        
        try:
            subject_data, electrodes_map = EEGLABIO.get_subject_data(self.file_paths[subject], is_epochs=False)
            self.update_subjects_metadata(subject, subject_data)
            
            for cond in self.conditions:   
                if cond not in subject_data:
                    continue
                
                else:
                    cond_data = subject_data[cond]
                    cond_elec_map = electrodes_map[cond]
                    FEUtilz.convert_electrode_names(cond_elec_map)    
                
                # slicing data for complexity and frequnecy features
                if (calc_complexity or calc_frequency) and self.electrodes:
                    sliced_data, updated_elec_map = FEUtilz.slice_epochs_2d(epochs_data=cond_data, sampling_rate=self.sampling_rate
                                                                            , start_time=np.inf, end_time=np.inf
                                                                            , time_offset=0, electrodes_map=cond_elec_map
                                                                            , electrode_names=self.electrodes)
                else:
                    sliced_data, updated_elec_map = cond_data, cond_elec_map
                
                # calculating Temporal Complexity features
                if calc_complexity:
                    complexity_table = self.complexity_analyzer.get_complexity_features(np.expand_dims(sliced_data, axis=0), updated_elec_map)
                else:
                    complexity_table = pd.DataFrame()
                    
                # calculating Frequency features
                if calc_frequency:
                    frequency_table = self.frequency_analyzer.get_frequency_features(sliced_data, updated_elec_map)
                else:
                    frequency_table = pd.DataFrame()
                
                curr_feature_table = pd.concat([complexity_table, frequency_table])
                curr_feature_table['Condition'] = cond
                curr_feature_table['Subject'] = subject
                    
                # Caching dataframe
                suject_feature_tables.append(curr_feature_table)
            
        except Exception as e:
            warnings.warn(f"***Problem with {subject}, continued to the next.")
        
        return suject_feature_tables if len(suject_feature_tables) > 0 else []
    
    def get_feature_table(self, calc_complexity: bool = True, calc_frequency: bool = True) -> pd.DataFrame:
        """
        Getting a full feature tables from all categories.
        :param calc_complexity: specify if temporal complexity features should be calculated.
        :param calc_frequency: specify if fooof-based frequency features should be calculated.
        :return: a dataframe with the following fields - Subject, Condition, Electrode, FeatureName, FeatureCategory, RawValue,
        where each row corresponds to a single feature, within a single condition, for the given subject data.
        -- Subject is the subject's ID as was extracted from the raw data file name.
        -- Condition is the experimental condition name.
        -- Electrode is the electrode from which the feature was extracted.
        -- FeatureCategory will start with 'Temporal_Complexity' followed by the ERP component name.
        -- FeatureName describes specifications of the extracted feature.
        -- RawValue is the actual feature value.
        -- NormalValue is the raw feature value after a softmax transformation.
        """
        all_feature_tables = []
        self.reset_subjects_metadata()
        self.frequency_analyzer.reset_accuracy()
        
        subjects = self.file_paths.keys()
        
        all_feature_tables = []
        for subject in tqdm(subjects, desc='Computing features per subject'):
            subject_feature_table = self.get_subject_feaures(subject, calc_complexity, calc_frequency)
            all_feature_tables.extend(subject_feature_table)
        
        # concatenating data frames
        feature_table = pd.concat(all_feature_tables)
        feature_table.drop_duplicates(ignore_index=True, inplace=True)
        FEUtilz.normalize(feature_table)
        FEUtilz.is_healthy(feature_table)
        
        FEUtilz.store_table(feature_table, 
                            file_name='resting_state_feature_table', 
                            results_dir=self.results_dir
                            )
                
        FEUtilz.store_table(pd.DataFrame(self.subjects_metadata), 
                            file_name='subjects_metadata_resting_state', 
                            results_dir=self.results_dir
                            )
        
        FEUtilz.store_table(self.frequency_analyzer.fooof_accuracy.reset_index(drop=True), 
                            file_name='fooof_accuracy_resting_state', 
                            results_dir=self.results_dir
                            )
        
        return feature_table
        
    def store_params(self):
        """
        Stores all the parameters used to calculate features into a single .txt file.
        """
        params = {
            'Complexity': {}
            , 'Frequency': {}
        }
        
        FEUtilz.store_params_per_category(self.complexity_analyzer, 'Complexity', params)
        FEUtilz.store_params_per_category(self.frequency_analyzer, 'Frequency', params)
        
        timestamp = dt.datetime.now().strftime('%d_%m_%Y_%H_%M')
        path = os.path.join(self.results_dir, rf"resting_state_model_parameters_{timestamp}.txt")
        with open(path, "w") as file:
            for category, data in params.items():
                file.write(f"----{category}:\n")
                for attr_name, attr in data.items():
                    file.write(f"{attr_name}: {attr}\n")
                file.write(f"\n")
        
        print(f'saved file at model_parameters_{timestamp}.txt')
