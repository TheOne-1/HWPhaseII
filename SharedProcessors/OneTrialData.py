from const import PROCESSED_DATA_PATH, MOCAP_SAMPLE_RATE, ROTATION_VIA_STATIC_CALIBRATION, \
    SPECIFIC_CALI_MATRIX, TRIAL_START_BUFFER, FILTER_WIN_LEN, SUB_NAMES, TRIAL_NAMES
import numpy as np
import pandas as pd
import xlrd


class OneTrialData:
    def __init__(self, subject_name, trial_name, readme_xls, sensor_sampling_fre, side, static_data_df=None):
        self._subject_name = subject_name
        self._subject_id = SUB_NAMES.index(subject_name)
        self._trial_name = trial_name
        self._side = side
        self._sensor_sampling_fre = sensor_sampling_fre
        self._static_data_df = static_data_df
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            # self._side = 'l'       # 'l' or 'r'
            data_folder = '\\200Hz\\'
        else:
            # self._side = 'r'       # 'l' or 'r'
            data_folder = '\\100Hz\\'
        # initialize the dataframe of gait data, including force plate, marker and IMU data
        gait_data_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + trial_name + '.csv'
        self.gait_data_df = pd.read_csv(gait_data_path, index_col=False)
        # initialize the dataframe of gait parameters, including loading rate, strike index, ...
        gait_param_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + 'param_of_' + trial_name + '.csv'
        buffer_sample_num = self._sensor_sampling_fre * TRIAL_START_BUFFER
        self.gait_data_df = self.gait_data_df.loc[buffer_sample_num:, :]        # skip the first several hundred data
        if static_data_df is not None:
            self.gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            self.gait_param_df = self.gait_param_df.loc[buffer_sample_num:, :]

        self.__init_subtrial_ends(readme_xls)

    def __init_subtrial_ends(self, readme_xls):
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        trial_id = TRIAL_NAMES.index(self._trial_name)
        if 'FPA' in self._trial_name or 'static trunk' in self._trial_name:
            self.__subtrial_ends = readme_sheet.row_values(trial_id + 2)[6:11]
        elif 'trunk' in self._trial_name:
            self.__subtrial_ends = readme_sheet.row_values(trial_id + 2)[6:12]
        else:
            self.__subtrial_ends = None
        if self.__subtrial_ends is not None:
            self.__subtrial_ends = \
                [int(element / MOCAP_SAMPLE_RATE * self._sensor_sampling_fre) for element in self.__subtrial_ends]

    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        data = self.gait_data_df[column_names].values
        if ROTATION_VIA_STATIC_CALIBRATION:
            data_rotated = np.zeros(data.shape)
            if self._subject_name in SPECIFIC_CALI_MATRIX.keys() and\
                    IMU_location in SPECIFIC_CALI_MATRIX[self._subject_name].keys():
                    dcm_mat = SPECIFIC_CALI_MATRIX[self._subject_name][IMU_location]
            else:
                dcm_mat = self.get_rotation_via_static_cali(IMU_location)
            data_len = data.shape[0]
            for i_sample in range(data_len):
                if acc:
                    data_rotated[i_sample, 0:3] = np.matmul(dcm_mat, data[i_sample, 0:3])
                if gyr:
                    data_rotated[i_sample, 3:6] = np.matmul(dcm_mat, data[i_sample, 3:6])
                if mag:
                    data_rotated[i_sample, 6:9] = np.matmul(dcm_mat, data[i_sample, 6:9])
            return data_rotated
        else:
            return data

    def get_rotation_via_static_cali(self, IMU_location):
        axis_name_gravity = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        data_gravity = self._static_data_df[axis_name_gravity]
        vector_gravity = np.mean(data_gravity.values, axis=0)

        axis_name_mag = [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        try:
            data_mag = self._static_data_df[axis_name_mag]
        except KeyError:
            pass
        vector_mag = np.mean(data_mag.values, axis=0)

        vector_2 = vector_gravity / np.linalg.norm(vector_gravity)
        vector_0 = np.cross(vector_mag, vector_gravity)
        vector_0 = vector_0 / np.linalg.norm(vector_0)
        vector_1 = np.cross(vector_2, vector_0)
        vector_1 = vector_1 / np.linalg.norm(vector_1)

        dcm_mat = np.array([vector_0, vector_1, vector_2])
        return dcm_mat

    def get_data_by_sample(self, IMU_location, param_name, acc=True, gyr=True, mag=False):
        if param_name == 'FPA':
            param_data = self.gait_param_df[self._side + '_' + param_name].values
        else:
            param_data = self.gait_param_df[param_name].values

        IMU_data = self.get_one_IMU_data(IMU_location, acc, gyr, mag)

        subtrial_array = np.zeros([IMU_data.shape[0]])
        if self.__subtrial_ends is not None:
            subtrial_ends_sorted = self.__subtrial_ends[:]
            subtrial_ends_sorted.sort()
            subtrial_start = 0
            for subtrial_end in subtrial_ends_sorted:
                subtrial_id = self.__subtrial_ends.index(subtrial_end)
                subtrial_array[subtrial_start:subtrial_end+1] = subtrial_id
                subtrial_start = subtrial_end
        return IMU_data, param_data, subtrial_array

    def get_data_by_sample_with_strike_off(self, IMU_location, param_name, acc=True, gyr=True, mag=False):
        if 'FPA' in param_name:
            param_data = self.gait_param_df[self._side + '_' + param_name].values
        else:
            param_data = self.gait_param_df[param_name].values

        IMU_data = self.get_one_IMU_data(IMU_location, acc, gyr, mag)
        strike_off_data = self.gait_param_df[['strikes_IMU_lfilter', 'offs_IMU_lfilter']]
        input_data = np.column_stack([IMU_data, strike_off_data])
        subtrial_array = np.zeros([IMU_data.shape[0]])
        if self.__subtrial_ends is not None:
            subtrial_ends_sorted = self.__subtrial_ends[:]
            subtrial_ends_sorted.sort()
            subtrial_start = 0
            for subtrial_end in subtrial_ends_sorted:
                subtrial_id = self.__subtrial_ends.index(subtrial_end)
                subtrial_array[subtrial_start:subtrial_end+1] = subtrial_id
                subtrial_start = subtrial_end
        return input_data, param_data, subtrial_array

    def get_data_by_step(self, IMU_location, param_name, acc=True, gyr=True, mag=False):
        # 重新写这个函数，重点是要IMU strike/off 和 Vicon strike/off分离，在IMU 数据附近找对应的vicon数据
        """
        GRFz: from strike to off
        acc and gyr: from off to off because information before strike might be useful
        """
        filter_delay = int(FILTER_WIN_LEN / 2)
        offs, step_num = self.get_offs()
        strikes, step_num = self.get_strikes()
        offs_imu, strikes_imu, step_num = self.get_offs_strikes_from_IMU()
        param_data = self.gait_param_df[param_name].values
        IMU_data = self.get_one_IMU_data(IMU_location, acc, gyr, mag)
        step_param_data, step_imu_data = [], []
        for i_step in range(step_num):
            strike_in_between = strikes[offs[i_step] < strikes]
            strike_in_between = strike_in_between[strike_in_between < offs[i_step+1]]
            if len(strike_in_between) != 1:
                continue
            step_start = offs[i_step] - filter_delay
            step_end = offs[i_step + 1] - filter_delay

            strikes_array = np.zeros([step_end - step_start, 1])
            strikes_array[strike_in_between - offs[i_step], 0] = 1
            # skip this step if the step_end exceeds the maximum data length
            if step_end > param_data.shape[0]:
                continue

            step_input = np.column_stack([IMU_data[step_start:step_end, :], strikes_array])
            step_imu_data.append(step_input)
            step_param_data.append(lr_data[step_start:step_end])
        step_imu_data, step_param_data = self.check_step_input_output(step_imu_data, step_param_data)
        return step_imu_data, step_param_data

    def get_strikes(self):
        strike_column = self._side + '_strikes'
        heel_strikes = self.gait_param_df[strike_column]
        strikes = np.where(heel_strikes == 1)[0]
        step_num = len(strikes) - 1
        return strikes, step_num

    def get_offs(self):
        off_column = self._side + '_offs'
        offs = self.gait_param_df[off_column]
        offs = np.where(offs == 1)[0]
        step_num = len(offs) - 1
        return offs, step_num

    def get_offs_strikes_from_IMU(self):
        """
        There is no side because by default 100Hz is the right side, 200Hz is the left side.
        :return:
        """
        off_column = 'offs_IMU_lfilter'
        offs = self.gait_param_df[off_column]
        offs = np.where(offs == 1)[0]
        step_num = len(offs) - 1

        strike_column = 'strikes_IMU_lfilter'
        strikes = self.gait_param_df[strike_column]
        strikes = np.where(strikes == 1)[0]
        return offs, strikes, step_num

    @staticmethod
    def check_step_data(step_data, up_diff_ratio=0.4, down_diff_ratio=0.3):        # check if step length is correct
        step_num = len(step_data)
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = len(step_data[i_step])
        step_len_mean = np.mean(step_lens)
        acceptable_len_max = step_len_mean * (1+up_diff_ratio)
        acceptable_len_min = step_len_mean * (1-down_diff_ratio)
        step_data_new = []
        for i_step in range(step_num):
            if acceptable_len_min < step_lens[i_step] < acceptable_len_max:
                step_data_new.append(step_data[i_step])
        return step_data_new

    @staticmethod
    def check_step_input_output(step_input, step_output, up_diff_ratio=0.4, down_diff_ratio=0.3):
        step_num = len(step_input)
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = len(step_input[i_step])
        step_len_mean = np.mean(step_lens)
        acceptable_len_max = step_len_mean * (1+up_diff_ratio)
        acceptable_len_min = step_len_mean * (1-down_diff_ratio)
        step_input_new, step_output_new = [], []
        for i_step in range(step_num):
            if acceptable_len_min < step_lens[i_step] < acceptable_len_max:
                step_input_new.append(step_input[i_step])
                step_output_new.append(step_output[i_step])
        return step_input_new, step_output_new


class OneTrialDataStatic(OneTrialData):
    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        data = self.gait_data_df[column_names]
        return data


