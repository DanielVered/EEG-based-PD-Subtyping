import os
import mne
import numpy as np

mne.set_log_level('ERROR')

class EEGLABIO:
    def __init__(self, data_path: str, conditions: list[str], subject_id_len: int = 6):
        """
        Creates an EpochsIO object to deal with IO operations of EEG data files.
        :param data_path: a path to the data directory with the EEG data. 
        Assumes raw files are grouped into distinct folders according to conditions names.
        :param conditions: a list of names of the experimental conditions.
        :param subject_id_len: the length of the SubjectID as the prefix of each file name.
        Default is subject_id_len = 7.
        """
        self.path = data_path
        self.conditions = conditions
        self.subject_id_len = subject_id_len
    
    @staticmethod
    def get_single_file_epoch(file_path: str) -> mne.Epochs:
        """
        Imports a single .set epochs-EEG data file.
        :param file_path: the path to a .set EEG file.
        """
        return mne.io.read_epochs_eeglab(file_path, montage_units='cm')

    @staticmethod
    def get_single_file_eeg(file_path: str) -> mne.io.Raw:
        """
        Imports a single .set EEG data file.
        :param file_path: the path to a .set EEG file.
        """
        return mne.io.read_raw_eeglab(file_path, montage_units='cm')
    
    def get_file_paths_by_subject(self) -> dict[str: dict[str: str]]:
        """
        Builds a dictionary of all files paths inside the given directory divided to conditions. 
        Assumes file name starts with subjectID.
        """
        paths = {}
        for cond in self.conditions:
            cond_path = os.path.join(self.path, cond)
            file_names = os.listdir(cond_path)
            for file in file_names:
                if file.endswith('.set'):
                    subject_id = file[:self.subject_id_len]
                    file_path = os.path.join(cond_path, file)
                    subject_paths = paths.get(subject_id, {})
                    subject_paths.update({cond: file_path})
                    paths.update({subject_id: subject_paths})
        return paths
    
    @staticmethod
    def get_subject_eeglab(subject_file_paths: dict[str: list[str]], is_epochs: bool) -> dict[str, [mne.io.Raw, mne.Epochs]]:
        """
        Get a subject specific eeg object from raw eeglab .set files.
        :param subject_file_paths: a dictionary with conditions as keys and paths for .set files as values.
        :param is_epochs: specify is the raw eeglab consists of epochs or not.
        :return subject_eeglab: a dictionary {condition: eeglab} representing the eeglab object of the given subject
        per condition.
        """
        subject_eeglab = {}
        for cond, path in subject_file_paths.items():
            if is_epochs:
                eeglab_object = EEGLABIO.get_single_file_epoch(path)
            else:
                eeglab_object = EEGLABIO.get_single_file_eeg(path)
            subject_eeglab.update({cond: eeglab_object})
        
        return subject_eeglab
    
    @staticmethod
    def get_subject_data(subject_file_paths: dict[str: list[str]], is_epochs: bool) -> tuple[dict]:
        """
        Get a subject specific data from raw files.
        :param subject: a dictionary with conditions as keys and paths for .set files as values.
        :param is_epochs: specify is the raw eeglab consists of epochs or not.
        :return subject_data: a dictionary {condition: data} representing the raw data of the given subject per condition.
        :return electrodes_map: a dictionary {condition: electrodes_map} representing the electrode names of the given subject per condition.
        """
        subject_epochs = EEGLABIO.get_subject_eeglab(subject_file_paths, is_epochs)
        subject_data = {cond: epochs.get_data() for cond, epochs in subject_epochs.items()}
        electrodes_map = {cond: epochs.ch_names for cond, epochs in subject_epochs.items()}
        
        return subject_data, electrodes_map
