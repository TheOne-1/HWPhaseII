from const import PROCESSED_DATA_PATH, MOCAP_SAMPLE_RATE, RAW_DATA_PATH, SUB_NAMES, TRIAL_NAMES
from OneTrialData import OneTrialData, OneTrialDataStatic
from DataStruct import DataStructSample


class DataReader:

    def __init__(self, sub_and_trials, param_name, sensor_sampling_fre, side, IMU_location):
        self._sub_and_trials = sub_and_trials  # subject names and corresponding trials in a dict
        self._sub_names = sub_and_trials.keys()
        self._param_name = param_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._side = side
        self._IMU_location = IMU_location
        # initialize the dataframe of gait data, including force plate, marker and IMU data
        self.__gait_data_path = PROCESSED_DATA_PATH + '\\'

    def prepare_data_by_sample(self):
        all_data_struct = DataStructSample()
        for subject_name in self._sub_names:
            print('loading data of: ' + subject_name)
            readme_xls = RAW_DATA_PATH + subject_name + '\\readme\\readme_' + subject_name + '.xlsx'
            subject_id = SUB_NAMES.index(subject_name)
            static_trial = OneTrialDataStatic(subject_name, 'static', readme_xls, self._sensor_sampling_fre,
                                              self._side)
            static_data_df = static_trial.get_one_IMU_data(self._IMU_location, acc=True, mag=True)

            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                trial_id = TRIAL_NAMES.index(trial_name)
                trial_processor = OneTrialData(subject_name, trial_name, readme_xls, self._sensor_sampling_fre,
                                               self._side, static_data_df=static_data_df)
                trial_input, trial_output, subtrial_array = \
                    trial_processor.get_data_by_sample(self._IMU_location, self._param_name)
                all_data_struct.append(trial_input, trial_output, subject_id, trial_id, subtrial_array)
        return all_data_struct

    def prepare_data_by_with_strike_off(self):
        all_data_struct = DataStructSample(input_dim=8)
        for subject_name in self._sub_names:
            print('loading data of: ' + subject_name)
            readme_xls = RAW_DATA_PATH + subject_name + '\\readme\\readme_' + subject_name + '.xlsx'
            subject_id = SUB_NAMES.index(subject_name)
            static_trial = OneTrialDataStatic(subject_name, 'static', readme_xls, self._sensor_sampling_fre,
                                              self._side)
            static_data_df = static_trial.get_one_IMU_data(self._IMU_location, acc=True, mag=True)

            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                trial_id = TRIAL_NAMES.index(trial_name)
                trial_processor = OneTrialData(subject_name, trial_name, readme_xls, self._sensor_sampling_fre,
                                               self._side, static_data_df=static_data_df)
                trial_input, trial_output, subtrial_array = \
                    trial_processor.get_data_by_sample_with_strike_off(self._IMU_location, self._param_name)
                all_data_struct.append(trial_input, trial_output, subject_id, trial_id, subtrial_array)
        return all_data_struct

    def prepare_data_by_step(self):
        all_data_struct = DataStructSample()
        for subject_name in self._sub_names:
            print('loading data of: ' + subject_name)
            readme_xls = RAW_DATA_PATH + subject_name + '\\readme\\readme_' + subject_name + '.xlsx'
            subject_id = SUB_NAMES.index(subject_name)
            static_trial = OneTrialDataStatic(subject_name, 'static', readme_xls, self._sensor_sampling_fre,
                                              self._side)
            static_data_df = static_trial.get_one_IMU_data(self._IMU_location, acc=True, mag=True)

            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                trial_id = TRIAL_NAMES.index(trial_name)
                trial_processor = OneTrialData(subject_name, trial_name, readme_xls, self._sensor_sampling_fre,
                                               self._side, static_data_df=static_data_df)
                trial_input, trial_output, subtrial_array = \
                    trial_processor.get_data_by_step(self._IMU_location, self._param_name)
                all_data_struct.append(trial_input, trial_output, subject_id, trial_id, subtrial_array)
        return all_data_struct







