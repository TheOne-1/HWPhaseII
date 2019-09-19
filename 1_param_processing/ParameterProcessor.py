import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
from const import TRIAL_NAMES, PLATE_SAMPLE_RATE, MOCAP_SAMPLE_RATE, HAISHENG_SENSOR_SAMPLE_RATE, \
    FOOT_SENSOR_BROKEN_SUBS
import xlrd
from numpy.linalg import norm
from StrikeOffDetectorIMU import StrikeOffDetectorIMU, StrikeOffDetectorIMUFilter


class ParamProcessor:
    def __init__(self, sub_name, readme_xls, trials, plot_strike_off=False, check_steps=False,
                 initialize_100Hz=True, initialize_200Hz=True):
        self._sub_name = sub_name
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        self.__weight = readme_sheet.cell_value(17, 1)  # in kilos
        self.__height = readme_sheet.cell_value(18, 1)  # in meters
        self.__plot_strike_off = plot_strike_off
        self.__check_steps = check_steps
        self.__initialize_100Hz = initialize_100Hz
        self.__initialize_200Hz = initialize_200Hz

        self._trials = list(trials)
        # remove static trials
        if TRIAL_NAMES[0] in self._trials:
            self._trials.remove(TRIAL_NAMES[0])

    def start_initalization(self, path):
        print('\n' + self._sub_name)
        fre_100_path = path + '\\' + self._sub_name + '\\100Hz\\'
        fre_200_path = path + '\\' + self._sub_name + '\\200Hz\\'

        if self.__initialize_100Hz:
            self.static_data_df = pd.read_csv(fre_100_path + TRIAL_NAMES[0] + '.csv', index_col=False)
            for trial_name in self._trials:
                print('\n' + trial_name + ' trial')
                self._current_trial = trial_name
                # initialize 100 Hz parameter
                print('100Hz')
                self._current_fre = 100
                gait_data_100_df = pd.read_csv(fre_100_path + trial_name + '.csv', index_col=False)
                trial_param_df_100 = self.init_trial_params(gait_data_100_df, HAISHENG_SENSOR_SAMPLE_RATE)
                self.__save_data(fre_100_path, trial_name, trial_param_df_100)
            # plt.show()

        if self.__initialize_200Hz:
            self.static_data_df = pd.read_csv(fre_200_path + TRIAL_NAMES[0] + '.csv', index_col=False)
            for trial_name in self._trials:
                print('\n' + trial_name + ' trial')
                self._current_trial = trial_name
                # initialize 200 Hz parameter
                print('200Hz')
                self._current_fre = 200
                gait_data_200_df = pd.read_csv(fre_200_path + trial_name + '.csv', index_col=False)
                trial_param_df_200 = self.init_trial_params(gait_data_200_df, MOCAP_SAMPLE_RATE)
                self.__save_data(fre_200_path, trial_name, trial_param_df_200)
            # plt.show()

    @staticmethod
    def resample_steps(steps_1000, sample_fre):
        ratio = int(1000 / sample_fre)
        steps_resampled = []
        for step in steps_1000:
            step_resampled = [round(step[0] / ratio), round(step[1] / ratio)]
            steps_resampled.append(step_resampled)
        return steps_resampled

    def init_trial_params(self, gait_data_df, sensor_sampling_rate):
        # get strikes and offs
        l_strikes, l_offs = self.get_strike_off(gait_data_df, plate=1)
        r_strikes, r_offs = self.get_strike_off(gait_data_df, plate=2)
        self.check_strikes_offs(-gait_data_df['f_1_z'], l_strikes, l_offs, self._current_trial + '   left foot')
        self.check_strikes_offs(-gait_data_df['f_2_z'], r_strikes, r_offs, self._current_trial + '   right foot')

        # get steps
        l_steps = self.get_legal_steps(l_strikes, l_offs, 'l', gait_data_df=gait_data_df)
        r_steps = self.get_legal_steps(r_strikes, r_offs, 'r', gait_data_df=gait_data_df)

        # get FPA and trunk angles
        FPA_all = self.get_FPA_all(gait_data_df)  # FPA of all the samples
        trunk_ml_angle, trunk_ap_angle = self.get_trunk_angles(gait_data_df)
        # self.check_trunk_angles(trunk_ml_angle, trunk_ap_angle)
        param_data = np.column_stack([trunk_ap_angle, trunk_ml_angle, l_strikes, r_strikes, l_offs, r_offs, FPA_all])
        param_data_df = pd.DataFrame(param_data)
        param_data_df.columns = ['trunk_ap_angle', 'trunk_ml_angle', 'l_strikes', 'r_strikes', 'l_offs', 'r_offs',
                                 'l_FPA', 'r_FPA']
        param_data_df.insert(0, 'marker_frame', gait_data_df['marker_frame'])

        if self._sub_name not in FOOT_SENSOR_BROKEN_SUBS:
            # get strikes and offs from IMU data
            estimated_strikes_lfilter, estimated_offs_lfilter = self.get_strike_off_from_imu_lfilter(
                gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True,
                plot_the_strike_off=self.__plot_strike_off)
            param_data_df.insert(len(param_data_df.columns), 'strikes_IMU_lfilter', estimated_strikes_lfilter)
            param_data_df.insert(len(param_data_df.columns), 'offs_IMU_lfilter', estimated_offs_lfilter)
            l_FPA_steps = self.get_FPA_steps(gait_data_df, FPA_all[:, 0], l_steps)
            r_FPA_steps = self.get_FPA_steps(gait_data_df, FPA_all[:, 1], r_steps)
            self.insert_param_data(param_data_df, l_FPA_steps, 'l_FPA_steps')
            self.insert_param_data(param_data_df, r_FPA_steps, 'r_FPA_steps')

        return param_data_df

    @staticmethod
    def check_loading_rate_all(l_loading_rate):
        plt.figure()
        plt.plot(l_loading_rate)

    @staticmethod
    def get_strike_off(gait_data_df, plate, threshold=20, comparison_len=4):
        if plate == 1:
            force = gait_data_df[['f_1_x', 'f_1_y', 'f_1_z']].values
        elif plate == 2:
            force = gait_data_df[['f_2_x', 'f_2_y', 'f_2_z']].values
        else:
            raise ValueError('Wrong plate number')

        force_norm = norm(force, axis=1)
        data_len = force_norm.shape[0]
        strikes, offs = np.zeros(data_len, dtype=np.int8), np.zeros(data_len, dtype=np.int8)
        i_sample = 0
        # go to the first stance phase
        while i_sample < data_len and force_norm[i_sample] < 300:
            i_sample += 1
        swing_phase = False
        while i_sample < data_len - comparison_len:
            # for swing phase
            if swing_phase:
                while i_sample < data_len - comparison_len:
                    i_sample += 1
                    lower_than_threshold_num = len(
                        np.where(force_norm[i_sample:i_sample + comparison_len] < threshold)[0])
                    if lower_than_threshold_num >= round(0.8 * comparison_len):
                        continue
                    else:
                        strikes[i_sample + round(0.8 * comparison_len) - 1] = 1
                        swing_phase = False
                        break
            # for stance phase
            else:
                while i_sample < data_len and force_norm[i_sample] > 300:  # go to the next stance phase
                    i_sample += 1
                while i_sample < data_len - comparison_len:
                    i_sample += 1
                    lower_than_threshold_num = len(
                        np.where(force_norm[i_sample:i_sample + comparison_len] < threshold)[0])
                    if lower_than_threshold_num >= round(0.8 * comparison_len):
                        offs[i_sample + round(0.2 * comparison_len)] = 1
                        swing_phase = True
                        break
        return strikes, offs

    def get_strike_off_1000(self, gait_data_df, plate_data_1000, sensor_sampling_rate, threshold=20):
        force = plate_data_1000[['f_1_x', 'f_1_y', 'f_1_z']].values
        force_norm = norm(force, axis=1)
        strikes, offs = self.get_raw_strikes_offs(force_norm, threshold, comparison_len=20)
        self.check_strikes_offs(force_norm, strikes, offs)

        # distribute strikes offs to left and right foot
        data_len = len(strikes)
        ratio = sensor_sampling_rate / PLATE_SAMPLE_RATE

        l_strikes, r_strikes = np.zeros(data_len), np.zeros(data_len)
        l_offs, r_offs = np.zeros(data_len), np.zeros(data_len)
        for i_sample in range(data_len):
            if strikes[i_sample] == 1:
                l_heel_y = gait_data_df.loc[round(i_sample * ratio), 'LFCC_y']
                r_heel_y = gait_data_df.loc[round(i_sample * ratio), 'RFCC_y']
                if l_heel_y == 0 or r_heel_y == 0:
                    raise ValueError('Marker missing')
                if l_heel_y > r_heel_y:
                    l_strikes[i_sample] = 1
                else:
                    r_strikes[i_sample] = 1
            if offs[i_sample] == 1:
                l_heel_y = gait_data_df.loc[round(i_sample * ratio), 'LFCC_y']
                r_heel_y = gait_data_df.loc[round(i_sample * ratio), 'RFCC_y']
                if l_heel_y == 0 or r_heel_y == 0:
                    raise ValueError('Marker missing')
                if l_heel_y < r_heel_y:
                    l_offs[i_sample] = 1
                else:
                    r_offs[i_sample] = 1
        return l_strikes, r_strikes, l_offs, r_offs

    def check_strikes_offs(self, force_norm, strikes, offs, title=''):
        strike_indexes = np.where(strikes == 1)[0]
        off_indexes = np.where(offs == 1)[0]
        data_len = min(strike_indexes.shape[0], off_indexes.shape[0])

        # check strike off by checking if each strike is followed by a off
        strike_off_detection_flaw = False
        if strike_indexes[0] > off_indexes[0]:
            diffs_0 = np.array(strike_indexes[:data_len]) - np.array(off_indexes[:data_len])
            diffs_1 = np.array(strike_indexes[:data_len - 1]) - np.array(off_indexes[1:data_len])
        else:
            diffs_0 = np.array(off_indexes[:data_len]) - np.array(strike_indexes[:data_len])
            diffs_1 = np.array(off_indexes[:data_len - 1]) - np.array(strike_indexes[1:data_len])
        if np.min(diffs_0) < 0 or np.max(diffs_1) > 0:
            strike_off_detection_flaw = True

        try:
            if strike_off_detection_flaw:
                raise ValueError('For trial {trial_name}, strike off detection result are wrong.'.format(
                    trial_name=self._current_trial))
            if self.__plot_strike_off:
                raise ValueError
        except ValueError as value_error:
            if len(value_error.args) != 0:
                print(value_error.args[0])
            plt.figure()
            plt.plot(force_norm)
            plt.grid()
            plt.plot(strike_indexes, force_norm[strike_indexes], 'g*')
            plt.plot(off_indexes, force_norm[off_indexes], 'gx')
            plt.title(title)

    def get_trunk_angles(self, gait_data_df):
        C7 = gait_data_df[['C7_x', 'C7_y', 'C7_z']].values
        l_PSIS = gait_data_df[['LIPS_x', 'LIPS_y', 'LIPS_z']].values
        r_PSIS = gait_data_df[['RIPS_x', 'RIPS_y', 'RIPS_z']].values
        middle_PSIS = (l_PSIS + r_PSIS) / 2
        vertical_vector = C7 - middle_PSIS
        trunk_ml_angle = 180 / np.pi * np.arctan(vertical_vector[:, 0] / vertical_vector[:, 2])
        trunk_ap_angle = 180 / np.pi * np.arctan(vertical_vector[:, 1] / vertical_vector[:, 2])
        return trunk_ml_angle, trunk_ap_angle

    @staticmethod
    def check_trunk_angles(trunk_ml_angle, trunk_ap_angle):
        plt.figure()
        plt.plot(trunk_ml_angle)
        plt.title('trunk swag')
        plt.figure()
        plt.plot(trunk_ap_angle)
        plt.title('trunk inclination')

    @staticmethod
    def get_projected_points(p0, p1, p2):
        """
        Project one point on a line in a 2D space
        :param p0: Coordinates of the toe
        :param p1: Coordinates of the heel
        :param p2: Coordinates of the COP
        :return:
        """
        [a0, b0] = p0
        [a1, b1] = p1
        [a2, b2] = p2
        the_mat = np.matrix([[a0 - a1, b0 - b1],
                             [b1 - b0, a0 - a1]])
        the_array = np.array([a0 * a2 - a1 * a2 + b0 * b2 - b1 * b2, a0 * b1 - a1 * b1 - b0 * a1 + a1 * b1])
        projected_point = np.matmul(the_mat.I, the_array.T)
        return projected_point

    @staticmethod
    def __save_data(folder_path, trial_name, data_all_df):
        # save param data
        data_file_str = '{folder_path}\\param_of_{trial_name}.csv'.format(
            folder_path=folder_path, trial_name=trial_name)
        data_all_df.to_csv(data_file_str, index=False)

    def get_strike_off_from_imu(self, gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True,
                                plot_the_strike_off=False):
        if sensor_sampling_rate == HAISHENG_SENSOR_SAMPLE_RATE:
            my_detector = StrikeOffDetectorIMU(self._current_trial, gait_data_df, param_data_df, 'r_foot',
                                               HAISHENG_SENSOR_SAMPLE_RATE)
            strike_delay, off_delay = 4, 4  # delay from the peak
        elif sensor_sampling_rate == MOCAP_SAMPLE_RATE:
            my_detector = StrikeOffDetectorIMU(self._current_trial, gait_data_df, param_data_df, 'l_foot',
                                               MOCAP_SAMPLE_RATE)
            strike_delay, off_delay = 8, 6  # delay from the peak
        else:
            raise ValueError('Wrong sensor sampling rate value')
        estimated_strike_indexes, estimated_off_indexes = my_detector.get_walking_strike_off(strike_delay, off_delay)
        if plot_the_strike_off:
            my_detector.show_IMU_data_and_strike_off(estimated_strike_indexes, estimated_off_indexes)
        data_len = gait_data_df.shape[0]
        estimated_strikes, estimated_offs = np.zeros([data_len]), np.zeros([data_len])
        estimated_strikes[estimated_strike_indexes] = 1
        estimated_offs[estimated_off_indexes] = 1
        if check_strike_off:
            my_detector.true_esti_diff(estimated_strike_indexes, 'strikes')
            my_detector.true_esti_diff(estimated_off_indexes, 'offs')
        return estimated_strikes, estimated_offs

    def get_strike_off_from_imu_lfilter(self, gait_data_df, param_data_df, sensor_sampling_rate, check_strike_off=True,
                                        plot_the_strike_off=False):
        """
        In the filter, lfilter was used so there are delays in the detected events (about 50 samples)
        """
        if sensor_sampling_rate == HAISHENG_SENSOR_SAMPLE_RATE:
            my_detector = StrikeOffDetectorIMUFilter(self._current_trial, gait_data_df, param_data_df, 'r_foot',
                                                     HAISHENG_SENSOR_SAMPLE_RATE)
            strike_delay, off_delay = 6, 5  # delay from the peak
        elif sensor_sampling_rate == MOCAP_SAMPLE_RATE:
            my_detector = StrikeOffDetectorIMUFilter(self._current_trial, gait_data_df, param_data_df, 'l_foot',
                                                     MOCAP_SAMPLE_RATE)
            strike_delay, off_delay = 10, 10  # delay from the peak
        else:
            raise ValueError('Wrong sensor sampling rate value')
        estimated_strike_indexes, estimated_off_indexes = my_detector.get_walking_strike_off(strike_delay, off_delay)
        if plot_the_strike_off:
            my_detector.show_IMU_data_and_strike_off(estimated_strike_indexes, estimated_off_indexes)
        data_len = gait_data_df.shape[0]
        estimated_strikes, estimated_offs = np.zeros([data_len]), np.zeros([data_len])
        estimated_strikes[estimated_strike_indexes] = 1
        estimated_offs[estimated_off_indexes] = 1
        if check_strike_off:
            my_detector.true_esti_diff(estimated_strike_indexes, 'strikes')
            my_detector.true_esti_diff(estimated_off_indexes, 'offs')
        return estimated_strikes, estimated_offs

    # FPA of all the samples
    def get_FPA_all(self, gait_data_df):
        l_toe = gait_data_df[['LFM2_x', 'LFM2_y', 'LFM2_z']].values
        l_heel = gait_data_df[['LFCC_x', 'LFCC_y', 'LFCC_z']].values

        forward_vector = l_toe - l_heel
        left_FPAs = - 180 / np.pi * np.arctan2(forward_vector[:, 0], forward_vector[:, 1])

        r_toe = gait_data_df[['RFM2_x', 'RFM2_y', 'RFM2_z']].values
        r_heel = gait_data_df[['RFCC_x', 'RFCC_y', 'RFCC_z']].values

        forward_vector = r_toe - r_heel
        right_FPAs = 180 / np.pi * np.arctan2(forward_vector[:, 0], forward_vector[:, 1])

        return np.column_stack([left_FPAs, right_FPAs])

    def get_FPA_steps(self, gait_data_df, FPA_all, steps):
        FPA_steps = []
        for step in steps:
            sample_20_gait_phase = int(round(step[0] + 0.2 * (step[1] - step[0])))
            sample_80_gait_phase = int(round(step[0] + 0.8 * (step[1] - step[0])))
            FPA_step = np.mean(FPA_all[sample_20_gait_phase:sample_80_gait_phase])
            marker_frame = gait_data_df.loc[round((step[0] + step[1]) / 2), 'marker_frame']
            FPA_steps.append([FPA_step, marker_frame])
        return FPA_steps

    def get_legal_steps(self, strikes, offs, side, gait_data_df=None):
        """
            Sometimes subjects have their both feet on the ground so it is necessary to do step check.
        :param strikes:
        :param offs:
        :param side:
        :param gait_data_df:
        :return:
        """
        stance_phase_sample_thd_lower = 0.3 * self._current_fre
        stance_phase_sample_thd_higher = 1 * self._current_fre

        strike_tuple = np.where(strikes == 1)[0]
        off_tuple = np.where(offs == 1)[0]
        steps = []
        abandoned_step_num = 0

        for strike in strike_tuple:
            off = off_tuple[strike + stance_phase_sample_thd_lower < off_tuple]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:
                off = off[0]
                steps.append([strike, off])
            else:
                abandoned_step_num += 1

        print('For {side} foot steps, {step_num} steps abandonded'.format(side=side, step_num=abandoned_step_num))
        if self.__check_steps:
            plt.figure()
            if side == 'l':
                grf_z = gait_data_df['f_1_z'].values
            elif side == 'r':
                grf_z = gait_data_df['f_2_z'].values
            else:
                raise ValueError('Wrong side value')

            for step in steps:
                plt.plot(grf_z[step[0]:step[1]])
            plt.show()
        return steps

    def insert_param_data(self, gait_data_df, insert_list, column_name):
        data_len = gait_data_df.shape[0]
        insert_data = np.zeros([data_len])
        for item in insert_list:
            row_index = gait_data_df.index[gait_data_df['marker_frame'] == item[1]]
            if len(row_index) == 0:
                row_index = gait_data_df.index[gait_data_df['marker_frame'] == item[1] + 1]
            insert_data[row_index[0]] = item[0]
        gait_data_df.insert(len(gait_data_df.columns), column_name, insert_data)

    @staticmethod
    def __law_of_cosines(vector1, vector2):
        vector3 = vector1 - vector2
        num = inner1d(vector1, vector1) + \
              inner1d(vector2, vector2) - inner1d(vector3, vector3)
        den = 2 * np.sqrt(inner1d(vector1, vector1)) * np.sqrt(inner1d(vector2, vector2))
        return 180 / np.pi * np.arccos(num / den)


class TrunkStaticProcessor(ParamProcessor):
    def __init__(self, sub_name, readme_xls, plot_strike_off=False,
                 initialize_100Hz=True, initialize_200Hz=True):
        super().__init__(sub_name, readme_xls, trials=[], plot_strike_off=plot_strike_off,
                         initialize_100Hz=initialize_100Hz, initialize_200Hz=initialize_200Hz)
        self._trials = ['static trunk']

    def init_trial_params(self, gait_data_df, sensor_sampling_rate):
        trunk_ml_angle, trunk_ap_angle = self.get_trunk_angles(gait_data_df)
        # self.check_trunk_angles(trunk_ml_angle, trunk_ap_angle)
        param_data = np.column_stack([trunk_ml_angle, trunk_ap_angle])
        param_data_df = pd.DataFrame(param_data)
        param_data_df.columns = ['trunk_ml_angle', 'trunk_ap_angle']
        param_data_df.insert(0, 'marker_frame', gait_data_df['marker_frame'])
        return param_data_df
