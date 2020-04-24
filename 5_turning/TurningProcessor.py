import numpy as np
import matplotlib.pyplot as plt
from const import HAISHENG_SENSOR_SAMPLE_RATE, DATA_COLUMNS_IMU, FONT_SIZE, LINE_WIDTH, FONT_DICT, FONT_DICT_X_SMALL
import pandas as pd
from DataProcessorHS import HaishengSensorReaderNoInterpo
from DataProcessorHS import ParamInitializerHS
from ProcessorFPAHS import InitFPA
from ProcessorFPA import ProcessorFPA
from DataProcessorHS import StrikeOffDetectorIMU_HS_Overground
from numpy import cos, sin
import copy
from numpy.linalg import norm
from experiment_II_const import *
from PaperFigures import ErrorBarFigure
import os
import matplotlib.lines as lines


class TurningProcessor:
    def __init__(self, test_date):
        self.test_date = test_date
        # if not os.path.isdir('../5_turning/result_conclusion/' + test_date):
        #     os.makedirs('../5_turning/result_conclusion/' + test_date)
        # if not os.path.isdir('../5_turning/result_conclusion/' + test_date + '/plots'):
        #     os.makedirs('../5_turning/result_conclusion/' + test_date + '/plots')
        self.sub_name = None
        self._placement_offset = None
        self._placement_R_foot_sensor = None
        self.gait_path = None

    def get_trial_fpa(self, show_trial=True):
        """
        :param show_trial: if true, show the trial result of each subject to manually assign trial ids.
        :return:
        """
        # step_result_df = pd.DataFrame(columns=['sub_name', 'trial_id', 'FPA_true', 'FPA_estis', 'fpa_tbme'])
        # summary_result_df = pd.DataFrame()
        fpa_esti_sub_dict, fpa_tbme_sub_dict = {}, {}
        for i_sub in range(len(SUB_NAMES_TURNING)):
            sub_name = SUB_NAMES_TURNING[i_sub]
            self.init_sub_info(sub_name)

            data_df = self.load_hs_sensor_data(self.gait_path, placement_offset=PLACEMENT_OFFSET)
            walking_ends = self.find_walking_ends(data_df)
            estimated_strikes, estimated_offs = self.get_strike_off_from_imu(data_df, None, None, False)
            gait_param_df = pd.DataFrame(np.column_stack([estimated_strikes, estimated_offs]))
            gait_param_df.columns = ['strikes_IMU', 'offs_IMU']
            steps = ParamInitializerHS.get_legal_steps(estimated_strikes, estimated_offs, 'l', False)
            steps_of_trials_dict = self.get_steps_of_each_trial(walking_ends, steps)
            self.add_a_virtual_step(steps_of_trials_dict, estimated_offs)

            _, stance_phase_flag = InitFPA.initalize_steps_and_stance_phase(data_df, gait_param_df)
            euler_angles_esti = InitFPA.get_euler_angles_complementary_from_stance(
                data_df, stance_phase_flag)
            acc_IMU_rotated = InitFPA.get_rotated_acc(self._placement_R_foot_sensor, data_df, euler_angles_esti)

            if show_trial:
                plt.figure()
                plt.plot(-data_df['gyr_x'] * 10)
                plt.grid()
                ax = plt.gca()

            fpa_esti_trial_dict, fpa_tbme_trial_dict = {}, {}
            for i_trial in range(len(walking_ends)):
                trial_steps = steps_of_trials_dict[i_trial]
                trial_steps.reverse()

                fpa_esti = InitFPA.get_FPA_via_max_acc(acc_IMU_rotated, trial_steps, start_percent=0.6, end_percent=1.2)
                fpa_esti_trial_dict[i_trial] = fpa_esti[fpa_esti != 0]
                fpa_tbme = self.get_FPA_tbme_step(data_df, trial_steps)
                fpa_tbme_trial_dict[i_trial] = [fpa_tuple[0] for fpa_tuple in fpa_tbme]

                if show_trial:
                    ax.text(trial_steps[0][0], 100, i_trial, fontsize=FONT_SIZE)
                    i_step = 0
                    for fpa in fpa_tbme:
                        plt.plot(fpa[1], fpa[0], 'c^')
                        ax.text(fpa[1], -90, i_step, fontsize=10)
                        i_step += 1
                    for i_sample in range(len(fpa_esti)):
                        if fpa_esti[i_sample] != 0:
                            plt.plot(i_sample, fpa_esti[i_sample], 'b^')
                    for step in trial_steps:
                        plt.plot(step[0], 0, 'go')
                        plt.plot(step[1], 0, 'rx')
            if show_trial:
                plt.show()
            fpa_esti_sub_dict[sub_name] = fpa_esti_trial_dict
            fpa_tbme_sub_dict[sub_name] = fpa_tbme_trial_dict
        return fpa_esti_sub_dict, fpa_tbme_sub_dict

    @staticmethod
    def save_results(fpa_sub_dict, file_name):
        """Distribute the auto detected trials into three categories: baseline, walking start, and turning"""
        result_df = pd.DataFrame()
        for i_sub in range(len(SUB_NAMES_TURNING)):
            sub_name = SUB_NAMES_TURNING[i_sub]
            category_trial_loc = SUB_AND_TRIAL_ID_TURNING[sub_name]

            data_array = np.full([MAX_STEP_NUM, len(TRIAL_NAMES_TURNING) * BASELINE_TRIAl_NUM], 1000.0)
            trial_num_of_each_cate = [BASELINE_TRIAl_NUM, WALKING_START_TRIAl_NUM, TURNING_TRIAl_NUM]
            i_column = 0
            for i_trial_cate in range(len(TRIAL_NAMES_TURNING)):
                trial_locs = category_trial_loc[TRIAL_NAMES_TURNING[i_trial_cate]]
                trial_num = trial_num_of_each_cate[i_trial_cate]
                for i_trial in range(trial_num):
                    loc = trial_locs[i_trial]
                    fpa_trial = fpa_sub_dict[sub_name][loc]
                    step_num = len(fpa_trial)
                    data_array[:step_num, i_column] = fpa_trial
                    i_column += 1
            sub_df = pd.DataFrame(data_array)
            # sub_df.columns = STEP_RESULT_COLUMNS[1:]
            sub_df.insert(0, 'sub_name', sub_name)
            result_df = pd.concat([result_df, sub_df])
        result_df.columns = STEP_RESULT_COLUMNS
        result_df.to_csv('../5_turning/result_conclusion/' + file_name + '.csv', index=False)

    @staticmethod
    def analyze_result(file_name):
        result_df = pd.read_csv('../5_turning/result_conclusion/' + file_name + '.csv', index_col=False)
        walking_start_result, turning_result = [], []
        for i_sub in range(len(SUB_NAMES_TURNING)):
            sub_name = SUB_NAMES_TURNING[i_sub]
            # 0. calculate baseline FPA, take the average of 10 steps before the last step
            sub_df = result_df[result_df['sub_name'] == sub_name]
            baseline_means = []
            for i_trial in range(BASELINE_TRIAl_NUM):
                baseline_fpa_step = sub_df['baseline_' + str(i_trial)]
                baseline_fpa_step = baseline_fpa_step[baseline_fpa_step < 1000.0]
                baseline_means.append(np.mean(baseline_fpa_step[BASELINE_START_STEP:BASELINE_END_STEP]))
            baseline_mean_overall = np.mean(baseline_means)

            # 1. draw walking start result
            for i_trial in range(WALKING_START_TRIAl_NUM):
                walking_start_fpa_step = sub_df['walking_start_' + str(i_trial)]
                walking_start_fpa_step = walking_start_fpa_step[walking_start_fpa_step < 1000.0]
                walking_start_fpa_step -= baseline_mean_overall
                walking_start_result.append(walking_start_fpa_step[:WALKING_START_TOTAL_STEP])

            # 2. draw turning result
            for i_trial in range(TURNING_TRIAl_NUM):
                sub_turning_occurance = SUB_TURNING_OCCURANCE[sub_name][i_trial]
                turning_fpa_step = sub_df['turning_' + str(i_trial)]
                turning_fpa_step = turning_fpa_step[turning_fpa_step < 1000.0]
                turning_fpa_step -= baseline_mean_overall
                turning_result.append(turning_fpa_step[
                                      sub_turning_occurance - BEFORE_TURNING_STEP:sub_turning_occurance + AFTER_TRUNING_STEP])

        walking_start_result_array = np.column_stack(walking_start_result)
        turning_result_array = np.column_stack(turning_result)
        return walking_start_result_array, turning_result_array

    @staticmethod
    def analyze_result_new_protocol(file_name):
        result_df = pd.read_csv('../5_turning/result_conclusion/' + file_name + '.csv', index_col=False)
        walking_start_result, turning_result = [], []
        for i_sub in range(len(SUB_NAMES_TURNING)):
            sub_name = SUB_NAMES_TURNING[i_sub]
            # 0. calculate baseline FPA, take the average of 10 steps before the last step
            sub_df = result_df[result_df['sub_name'] == sub_name]
            baseline_means = []
            for i_trial in range(BASELINE_TRIAl_NUM):
                baseline_fpa_step = sub_df['baseline_' + str(i_trial)]
                baseline_fpa_step = baseline_fpa_step[baseline_fpa_step < 1000.0]
                baseline_means.append(np.mean(baseline_fpa_step[BASELINE_START_STEP:BASELINE_END_STEP]))
            baseline_mean_overall = np.mean(baseline_means)

            # 1. draw walking start result
            for i_trial in range(WALKING_START_TRIAl_NUM):
                walking_start_fpa_step = sub_df['walking_start_' + str(i_trial)]
                walking_start_fpa_step = walking_start_fpa_step[walking_start_fpa_step < 1000.0]
                walking_start_fpa_step -= baseline_mean_overall
                walking_start_result.append(walking_start_fpa_step[:WALKING_START_TOTAL_STEP])

            # 2. draw turning result
            for i_trial in range(TURNING_TRIAl_NUM):
                sub_turning_occurance = SUB_TURNING_OCCURANCE[sub_name][i_trial]
                turning_fpa_step = sub_df['turning_' + str(i_trial)]
                turning_fpa_step = turning_fpa_step[turning_fpa_step < 1000.0]
                turning_fpa_step -= baseline_mean_overall
                turning_result.append(turning_fpa_step[
                                      sub_turning_occurance - BEFORE_TURNING_STEP:sub_turning_occurance + AFTER_TRUNING_STEP])

        walking_start_result_array = np.column_stack(walking_start_result)
        turning_result_array = np.column_stack(turning_result)
        return walking_start_result_array, turning_result_array

    @staticmethod
    def draw_ebars(means, stds, cate_id_list, marker, x_locs=None, ecolor='black'):
        if x_locs == None:
            x_locs = range(len(cate_id_list))
        ebar, caplines, barlinecols = plt.errorbar(x_locs, means, stds, markersize=10,
                                                   capsize=0, color=ecolor, fmt=marker, lolims=True, uplims=True,
                                                   elinewidth=LINE_WIDTH)
        for i_cap in range(2):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(25)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    @staticmethod
    def draw_walking_start(walking_start_array_esti, walking_start_array_tbme):
        fpa_esti_mean = np.mean(walking_start_array_esti, axis=1)
        fpa_esti_std = np.std(walking_start_array_esti, axis=1)
        fpa_tbme_mean = np.mean(walking_start_array_tbme, axis=1)
        fpa_tbme_std = np.std(walking_start_array_tbme, axis=1)

        plt.figure(figsize=(10, 6))
        ErrorBarFigure.format_plot()

        plot_tbme, = plt.plot(range(WALKING_START_TOTAL_STEP), fpa_tbme_mean, color='slategray', linewidth=LINE_WIDTH)
        TurningProcessor.draw_ebars(fpa_tbme_mean, fpa_tbme_std, range(WALKING_START_TOTAL_STEP), 'o', ecolor='slategray')

        plot_esti, = plt.plot(range(WALKING_START_TOTAL_STEP), fpa_esti_mean, color='brown', linewidth=LINE_WIDTH)
        TurningProcessor.draw_ebars(fpa_esti_mean, fpa_esti_std, range(WALKING_START_TOTAL_STEP), 's', ecolor='brown')

        ax = plt.gca()
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_xticklabels(range(WALKING_START_TOTAL_STEP+1), fontdict=FONT_DICT)
        ax.set_ylim(-50, 50)
        ax.set_yticks(range(-50, 51, 20))
        ax.set_yticklabels(range(-50, 51, 20), fontdict=FONT_DICT)
        plt.legend([plot_esti, plot_tbme], ['Algorithm Proposed in This Paper', 'Algorithm Proposed in [xx]'], handlelength=2,
                   handleheight=1.3, bbox_to_anchor=(0.37, 0.85), ncol=1, fontsize=FONT_SIZE, frameon=False)
        ax.set_xlabel('Number of Steps After Walking Start', fontdict=FONT_DICT)
        ax.set_ylabel('Foot Progression Angle (deg)', fontdict=FONT_DICT)
        plt.tight_layout(rect=[0, 0, 1, 1])

    @staticmethod
    def draw_turning(turning_array_esti, turning_array_tbme):
        fpa_esti_mean = np.mean(turning_array_esti, axis=1)
        fpa_esti_std = np.std(turning_array_esti, axis=1)
        fpa_tbme_mean = np.mean(turning_array_tbme, axis=1)
        fpa_tbme_std = np.std(turning_array_tbme, axis=1)

        fig = plt.figure(figsize=(12, 6))
        ErrorBarFigure.format_plot()

        total_step_num = BEFORE_TURNING_STEP + AFTER_TRUNING_STEP
        plot_tbme, = plt.plot(range(total_step_num), fpa_tbme_mean, '--', color='darkolivegreen', linewidth=LINE_WIDTH)
        TurningProcessor.draw_ebars(fpa_tbme_mean, fpa_tbme_std, range(total_step_num), 'o', ecolor='darkolivegreen')

        plot_esti, = plt.plot(range(total_step_num), fpa_esti_mean, color='brown', linewidth=LINE_WIDTH)
        TurningProcessor.draw_ebars(fpa_esti_mean, fpa_esti_std, range(total_step_num), 's', ecolor='brown')

        # separate before and after turning
        l2 = lines.Line2D([0.36, 0.36], [0.01, 0.845], linestyle='--', transform=fig.transFigure, color='gray')
        fig.lines.extend([l2])

        ax = plt.gca()
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_xticks(range(BEFORE_TURNING_STEP+AFTER_TRUNING_STEP))
        x_tick_label = [x for x in range(BEFORE_TURNING_STEP, 0, -1)] + [x for x in range(1, AFTER_TRUNING_STEP+1)]
        ax.set_xticklabels(x_tick_label, fontdict=FONT_DICT)
        ax.set_ylim(-80, 30)
        ax.set_yticks(range(-80, 21, 20))
        ax.set_yticklabels(range(-80, 21, 20), fontdict=FONT_DICT)
        plt.legend([plot_esti, plot_tbme], ['Algorithm Proposed in This Paper', 'Algorithm Proposed in [xx]'], handlelength=2,
                   handleheight=1.3, bbox_to_anchor=(0.45, 0.8), ncol=1, fontsize=FONT_SIZE, frameon=False)
        ax.text(6.6, -95, 'Steps After Turning', fontdict=FONT_DICT)
        ax.text(-0.2, -95, 'Steps Before Turning', fontdict=FONT_DICT)
        ax.set_ylabel('Foot Progression Angle (deg)', fontdict=FONT_DICT)
        plt.tight_layout(rect=[0, 0, 1, 1])

    def init_sub_info(self, sub_name):
        self.sub_name = sub_name
        self._placement_offset = 0
        offset_rad = - np.deg2rad(self._placement_offset)
        self._placement_R_foot_sensor = np.array([
            [cos(offset_rad), sin(offset_rad), 0],
            [-sin(offset_rad), cos(offset_rad), 0],
            [0, 0, 1]])
        self.gait_path = DATA_PATH_TURNING + '\\' + sub_name + '.CSV'

    @staticmethod
    def load_hs_sensor_data(data_path, placement_offset):
        sensor_data_reader = HaishengSensorReaderNoInterpo(data_path)
        sensor_data_cols = ['sample'] + DATA_COLUMNS_IMU + ['FPA_tbme_average', 'FPA_tbme_raw']
        sensor_data_all = sensor_data_reader.data_raw_df[sensor_data_cols]
        sensor_data_all[['FPA_tbme_raw', 'FPA_tbme_average']] -= placement_offset
        sensor_data_all[['acc_x', 'acc_y', 'acc_z']] = 9.81 * sensor_data_all[['acc_x', 'acc_y', 'acc_z']]
        sensor_data_all = sensor_data_all.rename(columns={'sample': 'IMU_frame'})
        return sensor_data_all

    @staticmethod
    def find_walking_ends(data_df):
        """Find the walking end via a three second foot lift."""
        check_len = 2 * HAISHENG_SENSOR_SAMPLE_RATE  # 3 seconds
        skip_len = int(0.5 * HAISHENG_SENSOR_SAMPLE_RATE)  # 0.5 seconds

        data_len = data_df.shape[0]
        acc_data = data_df[['acc_x', 'acc_y', 'acc_z']].values
        acc_smoothed = np.zeros(acc_data.shape)
        acc_smoothed_y = acc_smoothed[:, 1]
        for i_axis in range(3):
            acc_smoothed[:, i_axis] = ProcessorFPA.smooth(acc_data[:, i_axis], 50, 'flat')

        walking_ends = []
        last_clip_is_end = False
        for i_pos in range(0, data_len, skip_len):
            if np.mean(acc_smoothed_y[i_pos:i_pos + check_len]) < -5:
                if not last_clip_is_end:
                    walking_ends.append(i_pos + check_len)
                    last_clip_is_end = True
            else:
                last_clip_is_end = False

        # plt.plot(acc_data[:, 1])
        # plt.plot(acc_smoothed[:, 1])
        # plt.plot(walking_ends, [0 for x in walking_ends], '*')
        # plt.grid()
        # plt.show()

        return walking_ends

    @staticmethod
    def get_steps_of_each_trial(walking_ends, steps):
        current_i_step = len(steps) - 1
        walking_bout_separation_criterion = 4 * HAISHENG_SENSOR_SAMPLE_RATE
        within_walking_criterion = 5 * HAISHENG_SENSOR_SAMPLE_RATE
        steps_of_trials_dict = {}
        i_trial = len(walking_ends) - 1
        walking_ends.reverse()
        for walking_end in walking_ends:
            steps_of_trial = []
            for i_step in range(current_i_step, 0, -1):
                if steps[i_step][1] - walking_end < within_walking_criterion:
                    # end of trial found, go to the trial start
                    steps_of_trial.append(steps[i_step])
                    for j_step in range(i_step, 1, -1):
                        if steps[j_step][1] - steps[j_step - 1][1] < walking_bout_separation_criterion:
                            steps_of_trial.append(steps[j_step - 1])
                        else:
                            break
                    break
            steps_of_trials_dict[i_trial] = steps_of_trial
            i_trial -= 1
            current_i_step = j_step
        return steps_of_trials_dict

    @staticmethod
    def get_FPA_tbme_step(gait_data_df, steps):
        FPA_tbmes = []
        # start from the second step, so that virtual step won't be calculated
        for i_step in range(1, len(steps)):
            current_step = steps[i_step]
            current_middle_sample = round((current_step[0] + current_step[1]) / 2)
            FPA_tbme = gait_data_df.loc[current_step[1], 'FPA_tbme_average']
            FPA_tbmes.append([FPA_tbme, current_middle_sample])
        return FPA_tbmes

    @staticmethod
    def get_strike_off_from_imu(gait_data_df, param_data_df, trial_name, check_strike_off=True,
                                plot_the_strike_off=False):
        my_detector = StrikeOffDetectorIMU_HS_Overground(trial_name, gait_data_df, param_data_df, 'l_foot',
                                                         HAISHENG_SENSOR_SAMPLE_RATE)
        strike_delay, off_delay = -5, 0  # delay from the peak
        fre = 6
        estimated_strike_indexes, estimated_off_indexes = my_detector.get_walking_strike_off(strike_delay, off_delay,
                                                                                             fre)
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

    @staticmethod
    def get_FPA_via_max_acc(acc_IMU_rotated, steps, start_percent, end_percent, span=40, beta=3):
        """Use the ratio of axis acceleration at the peak norm acc"""
        data_len = acc_IMU_rotated.shape[0]
        FPA_estis = np.zeros([data_len])

        acc_IMU_smoothed = copy.deepcopy(acc_IMU_rotated)

        for i_axis in range(2):
            acc_IMU_smoothed[:, i_axis] = ProcessorFPA.smooth(acc_IMU_rotated[:, i_axis], span, 'hanning')

        planar_acc_norm = norm(acc_IMU_smoothed[:, :2], axis=1)

        for i_step in range(len(steps) - 1):
            current_step, next_step = steps[i_step], steps[i_step + 1]
            swing_phase_len = next_step[0] - current_step[1]
            start_sample = int(round(swing_phase_len * start_percent))
            end_sample = int(round(swing_phase_len * end_percent))
            acc_clip = acc_IMU_smoothed[current_step[1] + start_sample:current_step[1] + end_sample, 0:2]
            acc_norm_clip = planar_acc_norm[current_step[1] + start_sample:current_step[1] + end_sample]
            acc_norm_max_arg = np.argmax(acc_norm_clip)

            # acc_max_x_arg = np.argmax(np.abs(acc_clip[:, 0]))
            acc_max_x = acc_clip[acc_norm_max_arg, 0]
            # acc_max_y_arg = np.argmax(np.abs(acc_clip[:, 1]))
            acc_max_y = acc_clip[acc_norm_max_arg, 1]

            the_FPA_esti = np.arctan2(-acc_max_x, -acc_max_y) * 180 / np.pi
            the_sample = int((next_step[0] + next_step[1]) / 2)
            FPA_estis[the_sample] = the_FPA_esti

            # !!!
            if (the_FPA_esti) > 50:
                plt.figure()
                plt.plot(acc_IMU_smoothed[:, 0])

                plt.figure()
                plt.plot(acc_clip[:, :2])
                plt.plot(acc_norm_clip)

                plt.show()

        # plt.show()
        return FPA_estis

    @staticmethod
    def add_a_virtual_step(steps_of_trials_dict, estimated_offs):
        """Since the new algorithm need the toe off of the last step to calculate the FPA, a virtual step is appended
        in front of the trial steps. The toe off of that step is valid."""
        off_locs = np.where(estimated_offs == 1)[0]
        for i_trial in steps_of_trials_dict.keys():
            trial_steps = steps_of_trials_dict[i_trial]
            first_strike = trial_steps[-1][0]
            first_off = off_locs[first_strike - 100 < off_locs]
            first_off = first_off[first_off < first_strike]
            if len(first_off) > 0:
                virtual_step = [first_off[-1] - 50, first_off[-1]]
                trial_steps.append(virtual_step)
