import numpy as np
from HaishengSensorReader import HaishengSensorReader
import pandas as pd
import matplotlib.pyplot as plt
from const import DATA_PATH_HS, SUB_AND_TRIALS_HS, DATA_COLUMNS_IMU, MARKERS_HS, SUB_SELECTED_SPEEDS, \
    TRIAL_NAMES_HS, HAISHENG_SENSOR_SAMPLE_RATE, COLUMN_NAMES_HAISHENG, SUB_TRIAL_NAME_SPECIAL,\
    SUB_TRIAL_NUM, VICON_FOLDER_2_SUB
import xlrd
import textract
import re
import c3d
from numpy.linalg import norm
from StrikeOffDetectorIMU import StrikeOffDetectorIMU
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal
from numpy import cos, sin
import os


class StrikeOffDetectorIMUHS(StrikeOffDetectorIMU):
    def get_IMU_data(self, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += ['acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += ['gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += ['mag_' + axis for axis in ['x', 'y', 'z']]
        return self._gait_data_df[column_names]

    def show_IMU_data_and_strike_off(self, estimated_strike_indexes, estimated_off_indexes, cut_off_fre_strike_off=6):
        """
        This function is used for giving ideas about
        :return:
        """
        side = self._IMU_location[0]
        true_strikes = self._param_data_df[side + '_strikes']
        true_strike_indexes = np.where(true_strikes == 1)[0]
        true_offs = self._param_data_df[side + '_offs']
        true_off_indexes = np.where(true_offs == 1)[0]

        gyr_data = self.get_IMU_data(acc=False, gyr=True).values
        gyr_x = -gyr_data[:, 0]
        gyr_x = self.data_filt_left(gyr_x, cut_off_fre_strike_off, self._sampling_fre)
        plt.figure()
        plt.title(self._trial_name + '   gyr_x')
        plt.plot(gyr_x)
        strike_plt_handle = plt.plot(true_strike_indexes, gyr_x[true_strike_indexes], 'g*')
        off_plt_handle = plt.plot(true_off_indexes, gyr_x[true_off_indexes], 'gx')
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, gyr_x[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, gyr_x[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle[0], off_plt_handle[0], strike_plt_handle_esti[0], off_plt_handle_esti[0]],
                   ['true_strikes', 'true_offs', 'estimated_strikes', 'estimated_offs'])

    def find_peak_max(self, data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = signal.find_peaks(data_clip, width=width, height=height, prominence=prominence)
        if len(peaks) == 0:
            return None
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]

    @staticmethod
    def data_filt_left(data, cut_off_fre, sampling_fre, filter_order=2):
        fre = cut_off_fre / (sampling_fre / 2)
        b, a = butter(filter_order, fre, 'lowpass')
        if len(data.shape) == 1:
            data_filt = signal.lfilter(b, a, data)
        else:
            data_filt = signal.lfilter(b, a, data, axis=0)
        return data_filt

    def get_walking_strike_off(self, strike_delay, off_delay, cut_off_fre_strike_off=6):
        off_gyr_thd = 2  # threshold the minimum peak of medio-lateral heel strike
        off_gyr_prominence = 2

        gyr_data = self.get_IMU_data(acc=False, gyr=True).values
        gyr_x_unfilt = - gyr_data[:, 0]
        gyr_x = self.data_filt_left(gyr_x_unfilt, cut_off_fre_strike_off, self._sampling_fre)

        data_len = gyr_data.shape[0]
        strike_list, off_list = [], []

        # find the first off. When it is found
        start_buffer = 5 * self._sampling_fre
        clip_len = 4 * self._sampling_fre
        for i_clip in range(5):
            clip_start = i_clip * clip_len + start_buffer
            clip_end = (i_clip + 1) * clip_len + start_buffer

            max_peak_index = self.find_peak_max(gyr_x[clip_start:clip_end], height=off_gyr_thd,
                                                prominence=off_gyr_prominence)
            if max_peak_index is not None:
                last_off = max_peak_index + off_delay + i_clip * clip_len + start_buffer
                break
        if 'last_off' not in locals():
            raise ValueError('First off not found.')

        # # !!!
        # plt.plot(gyr_x_unfilt)
        # plt.plot(gyr_x)
        # plt.show()

        # find strikes and offs
        check_win_len = int(1.5 * self._sampling_fre)           # find strike off within this range
        for i_sample in range(last_off+1, data_len):
            if i_sample - last_off > check_win_len:
                try:
                    peaks, properties = find_peaks(gyr_x[last_off:i_sample], height=off_gyr_thd, prominence=off_gyr_prominence)

                    peak_heights = properties['peak_heights']
                    first_peak = peaks[0]
                    max_index = np.argmax(peak_heights)
                    highest_peak = peaks[max_index]

                    strike_list.append(first_peak + last_off + strike_delay)
                    off_list.append(highest_peak + last_off + off_delay)
                    last_off = off_list[-1]
                except IndexError as e:
                    if 2e3 < i_sample < 1e4:
                        print(i_sample)
                    last_off = last_off + int(self._sampling_fre * 0.4)     # skip this step

        if strike_list[-1] > data_len:
            strike_list.pop()
        if off_list[-1] > data_len:
            off_list.pop()
        return strike_list, off_list


class ParamInitializerHS:
    def __init__(self, sub_folder, plot_strike_off=False, check_steps=False):
        self.sub_folder = sub_folder
        self.sub_name = sub_folder.split('_')[-1]
        self.__plot_strike_off = plot_strike_off
        self.__check_steps = check_steps
        self._current_fre = HAISHENG_SENSOR_SAMPLE_RATE

    def start_init(self):
        for trial_id in range(len(TRIAL_NAMES_HS)):
            print('\n' + TRIAL_NAMES_HS[trial_id] + ' trial')
            gait_data_path = DATA_PATH_HS + 'processed/' + self.sub_folder + '/' + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_data_df = pd.read_csv(gait_data_path, index_col=False)
            # get strikes and offs (force plate measurements)
            l_strikes, l_offs = self.get_strike_off(gait_data_df)

            self.check_strikes_offs(-gait_data_df['f_1_z'], l_strikes, l_offs, TRIAL_NAMES_HS[trial_id])
            # get steps
            l_steps = self.get_legal_steps(l_strikes, l_offs, 'l', self.__check_steps, gait_data_df=gait_data_df)
            # get FPA and trunk angles
            l_FPA_all = self.get_FPA_all(gait_data_df)  # FPA of all the samples
            param_data = np.column_stack([l_strikes, l_offs, l_FPA_all])
            param_data_df = pd.DataFrame(param_data)
            param_data_df.columns = ['l_strikes', 'l_offs', 'l_FPA_all']
            estimated_strikes, estimated_offs = ParamInitializerHS.get_strike_off_from_imu(gait_data_df, param_data_df, TRIAL_NAMES_HS[trial_id])

            #!!!
            plt.plot(gait_data_df['f_1_z']/100)

            param_data_df.insert(len(param_data_df.columns), 'strikes_IMU', estimated_strikes)
            param_data_df.insert(len(param_data_df.columns), 'offs_IMU', estimated_offs)
            param_data_df.insert(0, 'vicon_frame', gait_data_df['vicon_frame'])
            FPA_true = self.get_FPA_true(gait_data_df, l_FPA_all, l_steps)
            FPA_tbme = self.get_FPA_tbme_step(gait_data_df, l_steps)
            self.insert_param_data(param_data_df, FPA_true, 'FPA_true')
            self.insert_param_data(param_data_df, FPA_tbme, 'FPA_tbme')
            self.save_trial_param(param_data_df, trial_id)

    @staticmethod
    def get_strike_off_from_imu(gait_data_df, param_data_df, trial_name, check_strike_off=True,
                                plot_the_strike_off=True):
        my_detector = StrikeOffDetectorIMUHS(trial_name, gait_data_df, param_data_df, 'l_foot', HAISHENG_SENSOR_SAMPLE_RATE)
        strike_delay, off_delay = -3, 0   # delay from the peak
        fre = 6
        estimated_strike_indexes, estimated_off_indexes = my_detector.get_walking_strike_off(strike_delay, off_delay, fre)
        if plot_the_strike_off:
            my_detector.show_IMU_data_and_strike_off(estimated_strike_indexes, estimated_off_indexes, fre)
        data_len = gait_data_df.shape[0]
        estimated_strikes, estimated_offs = np.zeros([data_len]), np.zeros([data_len])
        estimated_strikes[estimated_strike_indexes] = 1
        estimated_offs[estimated_off_indexes] = 1
        if check_strike_off:
            my_detector.true_esti_diff(estimated_strike_indexes, 'strikes')
            my_detector.true_esti_diff(estimated_off_indexes, 'offs')
        return estimated_strikes, estimated_offs

    def insert_param_data(self, gait_data_df, insert_list, column_name):
        data_len = gait_data_df.shape[0]
        insert_data = np.zeros([data_len])
        for item in insert_list:
            row_index = gait_data_df.index[gait_data_df['vicon_frame'] == item[1]]
            if len(row_index) == 0:
                row_index = gait_data_df.index[gait_data_df['vicon_frame'] == item[1] + 1]
            insert_data[row_index[0]] = item[0]
        gait_data_df.insert(len(gait_data_df.columns), column_name, insert_data)

    # FPA of all the samples
    def get_FPA_all(self, gait_data_df):
        l_toe = gait_data_df[['l_toe_x', 'l_toe_y', 'l_toe_z']].values
        l_heel = gait_data_df[['l_heel_x', 'l_heel_y', 'l_heel_z']].values

        forward_vector = l_toe - l_heel
        left_FPAs = - 180 / np.pi * np.arctan2(forward_vector[:, 0], forward_vector[:, 1])
        return left_FPAs

    def get_FPA_tbme_step(self, gait_data_df, steps):
        FPA_tbmes = []
        for i_step in range(len(steps)):
            current_step = steps[i_step]
            current_middle_sample = round((current_step[0] + current_step[1]) / 2)
            marker_frame = gait_data_df.loc[current_middle_sample, 'vicon_frame']
            FPA_tbme = gait_data_df.loc[current_step[1], 'FPA_tbme_average']
            FPA_tbmes.append([FPA_tbme, marker_frame])
        return FPA_tbmes

    def get_FPA_true(self, gait_data_df, FPA_all, steps):
        FPA_true = []
        for step in steps:
            sample_20_gait_phase = int(round(step[0] + 0.2 * (step[1] - step[0])))
            sample_80_gait_phase = int(round(step[0] + 0.8 * (step[1] - step[0])))
            FPA_step = np.mean(FPA_all[sample_20_gait_phase:sample_80_gait_phase])
            middle_sample = round((step[0] + step[1]) / 2)
            marker_frame = gait_data_df.loc[middle_sample, 'vicon_frame']
            FPA_true.append([FPA_step, marker_frame])
        return FPA_true

    @staticmethod
    def get_legal_steps(strikes, offs, side, check_steps, gait_data_df=None):
        """
            Sometimes subjects have their both feet on the ground so it is necessary to do step check.
        :param strikes:
        :param offs:
        :param side:
        :param gait_data_df:
        :return:
        """
        stance_phase_sample_thd_lower = 0.3 * HAISHENG_SENSOR_SAMPLE_RATE
        stance_phase_sample_thd_higher = 1 * HAISHENG_SENSOR_SAMPLE_RATE

        strike_tuple = np.where(strikes == 1)[0]
        off_tuple = np.where(offs == 1)[0]
        steps = []
        abandoned_step_num = 0
        grf_z = gait_data_df['f_1_z'].values
        last_off = 0
        for strike in strike_tuple:
            if strike < last_off:
                continue
            off = off_tuple[strike + stance_phase_sample_thd_lower < off_tuple]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:
                off = off[0]
                steps.append([strike, off])
                last_off = off
            else:
                abandoned_step_num += 1

        print('For {side} foot steps, {step_num} steps abandonded'.format(side=side, step_num=abandoned_step_num))
        if check_steps:
            plt.figure()
            for step in steps:
                plt.plot(grf_z[step[0]:step[1]])
        return steps

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
                raise ValueError('Strike off detection result are wrong.')
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

    @staticmethod
    def get_strike_off(gait_data_df, threshold=20, comparison_len=4):
        force = gait_data_df[['f_1_x', 'f_1_y', 'f_1_z']].values
        # force = StrikeOffDetectorIMU.data_filt(force, 25, HAISHENG_SENSOR_SAMPLE_RATE, 2)
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

    def save_trial_param(self, param_data_df, trial_id):
        save_path = DATA_PATH_HS + 'processed/' + self.sub_folder + '/param_of_' + \
                    TRIAL_NAMES_HS[trial_id] + '.csv'
        param_data_df.to_csv(save_path, index=False)


class HaishengSensorReaderNoInterpo(HaishengSensorReader):
    def _get_raw_data(self):
        """
        Interpolation was used to fill the missing package
        :return:
        """
        data_raw_df = pd.read_csv(self._file, header=None)
        data_raw_df.columns = COLUMN_NAMES_HAISHENG + ['FPA_tbme_average', 'FPA_tbme_raw']
        sample_num_col = HAISHENG_SENSOR_SAMPLE_RATE * (
                60 * data_raw_df['minute'] + data_raw_df['second'] + (data_raw_df['millisecond'] - 10) / 1000)
        data_raw_df.insert(0, 'sample', sample_num_col.astype(int))
        return data_raw_df

    def _get_sensor_data_processed(self, raw_data_df):
        pass


class SyncParams:
    def __init__(self):
        self.sensor_sync_start = None
        self.sensor_sync_end = None
        self.vicon_sync_start = None
        self.vicon_sync_end = None
        self.walk_start_row = None
        self.walk_end_row = None


class DataInitializerHS:
    def __init__(self, sub_folder):
        self.sub_folder = sub_folder
        self.sub_name = sub_folder.split('_')[-1]
        self._placement_offset = DataInitializerHS.init_placement_offset(sub_folder)

    def start_init(self):
        if self.sub_folder not in SUB_TRIAL_NUM.keys():
            self.start_init_normal_sub()
        else:
            self.start_init_trial_separated()

    def start_init_trial_separated(self):
        trial_id_order = 0
        for trial_id_overall in SUB_AND_TRIALS_HS[self.sub_folder]:
            trial_id_time = SUB_TRIAL_NUM[self.sub_folder][trial_id_order]
            if self.sub_folder not in SUB_TRIAL_NAME_SPECIAL.keys():
                sensor_file_name = self.sub_name + '_l_34_' + str(trial_id_time)
            elif trial_id_time in SUB_TRIAL_NAME_SPECIAL[self.sub_folder].keys():
                sensor_file_name = SUB_TRIAL_NAME_SPECIAL[self.sub_folder][trial_id_time]
            else:
                sensor_file_name = SUB_TRIAL_NAME_SPECIAL[self.sub_folder][-1]

            sensor_data = self.init_sensor_data_sub(sensor_file_name)
            sync_params = self.init_sync_point(trial_id_time-1)
            force_data_df = self.init_force_data(trial_id_overall, sync_params)
            sensor_data_df = self.init_sensor_data_trial(sensor_data, sync_params)
            marker_data_df = self.init_marker_data(trial_id_overall, sync_params)
            trial_data_df = pd.concat([force_data_df, marker_data_df, sensor_data_df], axis=1, join='inner')
            self.save_trial_data(trial_data_df, trial_id_order)
            trial_id_order += 1

    def start_init_normal_sub(self):
        """
        trial_id_overall_min: For all subjects' trials, the trial id according to the experiment time.
        trial_id_time: For each subject's trials, the trial id according to the experiment time.
        trial_id_order: For each subject's trials, the trial id according to the universal order (normal, small in, ...)
        :return:
        """
        sensor_data = self.init_sensor_data_sub()
        trial_id_overall_min = min(SUB_AND_TRIALS_HS[self.sub_folder])
        trial_id_order = 0
        for trial_id_overall in SUB_AND_TRIALS_HS[self.sub_folder]:
            trial_id_time = trial_id_overall - trial_id_overall_min
            sync_params = self.init_sync_point(trial_id_time)
            force_data_df = self.init_force_data(trial_id_overall, sync_params)
            sensor_data_df = self.init_sensor_data_trial(sensor_data, sync_params)
            marker_data_df = self.init_marker_data(trial_id_overall, sync_params)
            trial_data_df = pd.concat([force_data_df, marker_data_df, sensor_data_df], axis=1, join='inner')
            self.save_trial_data(trial_data_df, trial_id_order)
            trial_id_order += 1


    @staticmethod
    def init_placement_offset(sub_folder):
        # step 0, get shoe sizes
        normal_trial_id = SUB_AND_TRIALS_HS[sub_folder][0]
        hs_processed_file = DATA_PATH_HS + 'sensor/' + sub_folder + '/Sensor_L_Trial' + str(normal_trial_id) + \
                            '_Processed_' + str(SUB_SELECTED_SPEEDS[sub_folder]) + 'ms_normal.xlsx'
        file = xlrd.open_workbook(hs_processed_file)
        table = file.sheets()[0]
        placement_offset = table.cell(1, 3).value
        return placement_offset

    def init_force_data(self, trial_id_overall, sync_params):
        if self.sub_folder in VICON_FOLDER_2_SUB:
            data_path = DATA_PATH_HS + 'vicon/s1_8/force/Trial' + str(trial_id_overall) + '.CSV'
        else:
            data_path = DATA_PATH_HS + 'vicon/s9_14/force/Trial' + str(trial_id_overall) + '.CSV'
        force_data = pd.read_csv(data_path, index_col=False, skiprows=range(5), header=None,
                                 usecols=[0, 2, 3, 4])
        force_data.columns = ['vicon_frame', 'f_1_x', 'f_1_y', 'f_1_z']

        # data filtering
        force_data_filtered = StrikeOffDetectorIMU.data_filt(force_data.iloc[:, 1:], 20, HAISHENG_SENSOR_SAMPLE_RATE, 2)
        force_data_filtered = pd.DataFrame(force_data_filtered)
        force_data_filtered.columns = ['f_1_x', 'f_1_y', 'f_1_z']
        force_data_filtered.insert(0, 'vicon_frame', force_data['vicon_frame'])

        force_data_clip = force_data_filtered.iloc[sync_params.vicon_sync_start:sync_params.vicon_sync_end, :]
        force_data_clip = force_data_clip.reset_index(drop=True)
        return force_data_clip

    def save_trial_data(self, trial_data_df, trial_id_order):
        save_path = DATA_PATH_HS + 'processed/' + self.sub_folder
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_file = save_path + '/' + TRIAL_NAMES_HS[trial_id_order] + '.csv'
        trial_data_df.to_csv(save_file, index=False)

    @staticmethod
    def init_sensor_data_trial(sensor_data, sync_params):
        # trial_index_0 = sync_params.sensor_sync_start < sensor_data['sample']
        # sensor_data = sensor_data[trial_index_0]
        sensor_data_clip = sensor_data.iloc[sync_params.sensor_sync_start+1:sync_params.sensor_sync_end+1]
        sensor_data_clip = sensor_data_clip.reset_index(drop=True)
        return sensor_data_clip

    def init_sync_point(self, i_th_trial):
        sync_params = SyncParams()
        sync_file_path = DATA_PATH_HS + 'syncpoint/' + self.sub_folder + '/L_34_' + str(i_th_trial+1) + '.docx'
        if not os.path.isfile(sync_file_path):
            sync_file_path = DATA_PATH_HS + 'syncpoint/' + self.sub_folder + '/34_L_' + str(i_th_trial+1) + '.docx'
        sync_str = str(textract.process(sync_file_path))
        sync_str_list = sync_str.split('\\n')
        sensor_sync_temp, vicon_sync_temp, walk_sync_temp = [], [], []
        for sync_row in sync_str_list:
            # find "Sensor_Sync_Time"
            if 'Sensor_Sync_Time' in sync_row:
                nums_in_the_row = re.findall(r'[1-9][0-9]*', sync_row)
                sensor_sync_temp.append(int(nums_in_the_row[1]))

            # find "Vicon_Sync_Time"
            elif 'Vicon_Sync_Time' in sync_row:
                nums_in_the_row = re.findall(r'[1-9][0-9]*', sync_row)
                vicon_sync_temp.append(int(nums_in_the_row[1]))

            # find walking frames
            elif '_Frame' in sync_row:
                nums_in_the_row = re.findall(r'[1-9][0-9]*', sync_row)
                walk_sync_temp.append(int(nums_in_the_row[0]))

        sync_params.sensor_sync_start = sensor_sync_temp[0]
        sync_params.sensor_sync_end = sensor_sync_temp[0] + vicon_sync_temp[1] - vicon_sync_temp[0]
        sync_params.vicon_sync_start = vicon_sync_temp[0]
        sync_params.vicon_sync_end = vicon_sync_temp[1]
        sync_params.walk_start_row = walk_sync_temp[0] - vicon_sync_temp[0]
        sync_params.walk_end_row = walk_sync_temp[1] - vicon_sync_temp[0]
        return sync_params

    def init_sensor_data_sub(self, file_name=None):
        if file_name is None:
            file_name = self.sub_name + '_l_34'
        data_path = DATA_PATH_HS + 'sensor/' + self.sub_folder + '/' + file_name + '.CSV'
        sensor_data_reader = HaishengSensorReaderNoInterpo(data_path)
        sensor_data_cols = ['sample'] + DATA_COLUMNS_IMU + ['FPA_tbme_average', 'FPA_tbme_raw']
        sensor_data_all = sensor_data_reader.data_raw_df[sensor_data_cols]
        sensor_data_all[['FPA_tbme_raw', 'FPA_tbme_average']] -= self._placement_offset
        sensor_data_all[['acc_x', 'acc_y', 'acc_z']] = 9.81 * sensor_data_all[['acc_x', 'acc_y', 'acc_z']]
        sensor_data_all = sensor_data_all.rename(columns={'sample': 'IMU_frame'})
        return sensor_data_all

    def init_marker_data(self, trial_id_overall, sync_params):
        row_num = sync_params.vicon_sync_end - sync_params.vicon_sync_start
        data_array = np.zeros([row_num, 3 * len(MARKERS_HS)])
        marker_num = len(MARKERS_HS)
        if self.sub_folder in VICON_FOLDER_2_SUB:
            data_path = DATA_PATH_HS + 'vicon/s1_8/c3d/Trial' + str(trial_id_overall) + '.c3d'
        else:
            data_path = DATA_PATH_HS + 'vicon/s9_14/c3d/Trial' + str(trial_id_overall) + '.c3d'
        with open(data_path, 'rb') as handle:
            reader = c3d.Reader(handle)
            marker_names_c3d = reader.point_labels
            for i_marker in range(marker_num):
                if MARKERS_HS[i_marker] not in marker_names_c3d[i_marker]:
                    raise ValueError('Wrong marker order')

            for i_vicon, points, _ in reader.read_frames():
                if sync_params.vicon_sync_start <= i_vicon < sync_params.vicon_sync_end:
                    i_processed = i_vicon - sync_params.vicon_sync_start
                    for i_marker in range(marker_num):
                        data_array[i_processed, i_marker*3:(1+i_marker)*3] = points[i_marker, :3]
        marker_column_names = [marker + axis for marker in MARKERS_HS for axis in ['_y', '_x', '_z']]

        # Change the x-axis direction. The left foot x coordinate should be smaller than that of the right foot.
        data_array[:, 1] = 1120 - data_array[:, 1]
        data_array[:, 4] = 1120 - data_array[:, 4]

        data_array_filtered = StrikeOffDetectorIMU.data_filt(data_array, 15, HAISHENG_SENSOR_SAMPLE_RATE, 2)
        marker_data_df = pd.DataFrame(data_array_filtered, columns=marker_column_names)
        marker_data_df = marker_data_df.reset_index(drop=True)
        return marker_data_df

