import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from const import DATA_PATH_HS, SUB_AND_TRIALS_HS, DATA_COLUMNS_IMU, MARKERS_HS, SUB_SELECTED_SPEEDS, \
    TRIAL_NAMES_HS, HAISHENG_SENSOR_SAMPLE_RATE, COLUMN_NAMES_HAISHENG, SUB_NAMES_HS
from numpy.linalg import norm
from StrikeOffDetectorIMU import StrikeOffDetectorIMU
from numpy import cos, sin
from ProcessorFPA import ProcessorFPA
from transforms3d.euler import euler2mat
from DataProcessorHS import DataInitializerHS
from Evaluation import Evaluation
from scipy.signal import butter, lfilter
from Processor import Processor
import os


class InitFPA:
    def __init__(self, test_date):
        self.test_date = test_date
        self.sub_folder = None
        self.sub_name = None
        self._placement_offset = None
        self._placement_R_foot_sensor = None
        self.base_path = None

    def start_init(self):
        step_result_df = pd.DataFrame(columns=['sub_name', 'trial_id', 'FPA_true', 'FPA_estis', 'FPA_tbme'])
        summary_result_df = pd.DataFrame()
        for i_sub in range(len(SUB_NAMES_HS)):
            sub_folder = SUB_NAMES_HS[i_sub]
            self.init_sub_info(sub_folder)
            step_result_df, summary_result_df = self.backward_fpa_estimation(step_result_df, summary_result_df)
        summary_result_df = self.format_result_summary(summary_result_df)
        self.save_result_df('../3_FPA_Haisheng/result_conclusion/step_result/step_result', step_result_df)
        self.save_result_df('../3_FPA_Haisheng/result_conclusion/summary/summary_result', summary_result_df)

    @staticmethod
    def format_result_summary(summary_result_df):
        summary_result_df.columns = ['sub_name', 'correlation', 'RMSE', 'mean_error']
        summary_result_df.loc[-1] = ['absolute mean', np.mean(summary_result_df['correlation']),
                                     np.mean(summary_result_df['RMSE']),
                                     np.mean(abs(summary_result_df['mean_error']))]
        return summary_result_df

    @staticmethod
    def save_result_df(base_path, predict_result_df):
        file_path = base_path + '.csv'
        i_file = 0
        while os.path.isfile(file_path):
            i_file += 1
            file_path = base_path + '_' + str(i_file) + '.csv'
        predict_result_df.to_csv(file_path, index=False)

    def init_sub_info(self, sub_folder):
        self.sub_folder = sub_folder
        self.sub_name = sub_folder.split('_')[-1]
        self._placement_offset = DataInitializerHS.init_placement_offset(sub_folder)
        offset_rad = - np.deg2rad(self._placement_offset)
        self._placement_R_foot_sensor = np.array([
            [cos(offset_rad), sin(offset_rad), 0],
            [-sin(offset_rad), cos(offset_rad), 0],
            [0, 0, 1]])
        self.base_path = DATA_PATH_HS + 'processed/' + sub_folder + '/'

    def show_step_acc(self):
        """
        Show the average step. All the steps were interpolated to 50 samples.
        :return:
        """
        acc_ori_list, acc_rota_list, steps_list = [], [], []
        for trial_id in range(len(TRIAL_NAMES_HS)):
            print(TRIAL_NAMES_HS[trial_id])
            self.current_trial = TRIAL_NAMES_HS[trial_id]

            gait_data_path = self.base_path + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_data_df = pd.read_csv(gait_data_path, index_col=False)

            gait_param_path = self.base_path + 'param_of_' + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(gait_param_df, stance_after_strike=10,
                                                                             stance_before_off=30)
            acc_ori = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values

            euler_angles_esti = self.get_euler_angles_gradient_decent_from_stance(
                gait_data_df, stance_phase_flag, base_correction_coeff=0.02)

            euler_angles_esti_1 = self.get_euler_angles_gradient_decent(
                gait_data_df, stance_phase_flag, base_correction_coeff=0.02)

            acc_IMU_rotated = self.get_rotated_acc(gait_data_df, euler_angles_esti, acc_cut_off_fre=None)

            # plt.plot(euler_angles_esti[:, 1])
            # plt.plot(euler_angles_esti_1[:, 1])
            # plt.plot(stance_phase_flag)
            # plt.show()

            acc_ori_list.append(acc_ori)
            acc_rota_list.append(acc_IMU_rotated)
            steps_list.append(steps)

        plt.figure(figsize=(15, 10))
        for trial_id in range(len(TRIAL_NAMES_HS)):
            acc_ori, acc_rota, steps = acc_ori_list[trial_id], acc_rota_list[trial_id], steps_list[trial_id]
            # figure of acc without "flatten"
            if trial_id < 4:
                plt.subplot(2, 4, trial_id + 1)
            else:
                plt.subplot(2, 4, trial_id + 2)
            plt.title(TRIAL_NAMES_HS[trial_id])
            acc_ori_step_array, acc_rota_step_array = np.zeros([len(steps) - 6, 50]), np.zeros([len(steps) - 6, 50])
            for i_step in range(3, len(steps) - 3):
                last_off = steps[i_step][1]
                current_strike = steps[i_step + 1][0]
                acc_ori_step_array[i_step - 3, :] = Processor.resample_channel(acc_ori[last_off:current_strike, 0], 50)
                acc_rota_step_array[i_step - 3, :] = Processor.resample_channel(acc_rota[last_off:current_strike, 0], 50)

            plt.plot(np.mean(acc_ori_step_array, axis=0))
            plt.plot(np.mean(acc_rota_step_array, axis=0))

        plt.show()

    def backward_fpa_estimation(self, step_result_df, summary_result_df):
        fpa_true_list, fpa_esti_list = [], []
        for trial_id in range(len(TRIAL_NAMES_HS)):
            print(TRIAL_NAMES_HS[trial_id])
            self.current_trial = TRIAL_NAMES_HS[trial_id]

            gait_data_path = self.base_path + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_data_df = pd.read_csv(gait_data_path, index_col=False)

            gait_param_path = self.base_path + 'param_of_' + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(gait_param_df, stance_after_strike=10,
                                                                             stance_before_off=30)
            euler_angles_esti = self.get_euler_angles_gradient_decent_from_stance(
                gait_data_df, stance_phase_flag, base_correction_coeff=0.02)
            euler_angles_esti_1 = self.get_euler_angles_gradient_decent(
                gait_data_df, stance_phase_flag, base_correction_coeff=0.02)

            FPA_true = gait_param_df['FPA_true'].values.reshape(-1, 1)

            # plt.figure()
            # plt.plot(euler_angles_esti[:, 0])
            # plt.plot(euler_angles_esti_1[:, 0])
            # plt.figure()
            # plt.plot(euler_angles_esti[:, 1])
            # plt.plot(euler_angles_esti_1[:, 1])
            # plt.show()

            acc_IMU_rotated = self.get_rotated_acc(gait_data_df, euler_angles_esti, acc_cut_off_fre=None)
            FPA_estis = self.get_FPA_via_acc_before_strike_ratio(acc_IMU_rotated, steps, win_before_off=50, win_after_off=20)
            FPA_tbme = gait_param_df['FPA_tbme']

            step_trial_result_df = pd.DataFrame(np.column_stack([FPA_true, FPA_estis, FPA_tbme]))
            step_trial_result_df.columns = ['FPA_true', 'FPA_estis', 'FPA_tbme']
            step_trial_result_df.insert(0, 'trial_id', trial_id)
            step_trial_result_df.insert(0, 'sub_name', self.sub_folder)
            step_result_df = step_result_df.append(step_trial_result_df)

            fpa_true_temp, fpa_esti_temp = ProcessorFPA.compare_result(FPA_estis, FPA_true, steps)
            fpa_true_list.extend(fpa_true_temp[3:-3])
            fpa_esti_list.extend(fpa_esti_temp[3:-3])
        self.plot_sub_result(fpa_true_list, fpa_esti_list)
        pearson_coeff, RMSE, mean_error = Evaluation._get_all_scores(
            np.array(fpa_true_list), np.array(fpa_esti_list), precision=3)
        summary_result_df = Evaluation.insert_prediction_result(
            summary_result_df, self.sub_folder, pearson_coeff, RMSE, mean_error)
        return step_result_df, summary_result_df

        # save the sub result

    def plot_sub_result(self, fpa_true_list, fpa_esti_list):
        # show the sub result
        plt.figure()
        fpa_true_array, fpa_esti_array = np.array(fpa_true_list), np.array(fpa_esti_list)
        plt.plot(fpa_true_array, fpa_esti_array, '.')
        plt.plot([-20, 60], [-20, 60], 'black')
        correlation_coeff, RMSE, mean_error = Evaluation._get_all_scores(fpa_true_array, fpa_esti_array, precision=2)
        plt.title(self.sub_folder + '   RMSE: ' + str(RMSE[0]) + '   mean error: ' + str(mean_error))


    def param_optimizer(self):
        # param_range = range(-20, 20)
        predict_result_df = pd.DataFrame()
        for trial_id in range(len(TRIAL_NAMES_HS)):
            print(TRIAL_NAMES_HS[trial_id])
            self.current_trial = TRIAL_NAMES_HS[trial_id]

            gait_data_path = self.base_path + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_data_df = pd.read_csv(gait_data_path, index_col=False)

            gait_param_path = self.base_path + 'param_of_' + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(gait_param_df, stance_after_strike=10,
                                                                             stance_before_off=30)
            euler_angles_esti = self.get_euler_angles_gradient_decent_from_stance(
                gait_data_df, stance_phase_flag, base_correction_coeff=0.02)
            euler_angles_esti_1 = self.get_euler_angles_gradient_decent(
                gait_data_df, stance_phase_flag, base_correction_coeff=0.02)

            output_data = gait_param_df['FPA_true'].values.reshape(-1, 1)

            # plt.figure()
            # plt.plot(euler_angles_esti[:, 0])
            # plt.plot(euler_angles_esti_1[:, 0])
            # plt.figure()
            # plt.plot(euler_angles_esti[:, 1])
            # plt.plot(euler_angles_esti_1[:, 1])
            # plt.show()

            acc_IMU_rotated = self.get_rotated_acc(gait_data_df, euler_angles_esti, acc_cut_off_fre=None)
            FPA_estis = self.get_FPA_via_acc_before_strike_ratio(acc_IMU_rotated, steps, win_before_off=50, win_after_off=20)
            # FPA_estis = gait_param_df['FPA_tbme'].values
            fpa_true_temp, fpa_esti_temp = ProcessorFPA.compare_result(FPA_estis, output_data, steps)
            fpa_true_list.extend(fpa_true_temp[5:-5])
            fpa_esti_list.extend(fpa_esti_temp[5:-5])

            # plt.figure()
            # plt.plot(fpa_true_temp[5:-5], fpa_esti_temp[5:-5], '.')
            # plt.plot([-20, 60], [-20, 60], 'black')
            # plt.show()


        self.plot_sub_result(fpa_true_list, fpa_esti_list)


    @staticmethod
    def get_euler_angles_gradient_decent_from_stance(gait_data_df, stance_phase_flag, base_correction_coeff=0.01):
        """
        start initialization from the end of each stance phase
        :return:
        """
        delta_t = 1 / HAISHENG_SENSOR_SAMPLE_RATE

        acc_IMU = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values
        gyr_IMU = gait_data_df[['gyr_x', 'gyr_y', 'gyr_z']].values
        data_len = gait_data_df.shape[0]

        # gyr_IMU_moved = np.zeros(gyr_IMU.shape)
        # gyr_IMU_moved[:-1, :] = gyr_IMU[1:, :]
        angle_augments = gyr_IMU * delta_t
        euler_angles_esti = np.zeros([data_len, 3])

        last_stance_end = 0     # the end of gyr integration

        for i_sample in range(data_len):
            """1. initialize at the end of stance phase"""
            if stance_phase_flag[i_sample] and not stance_phase_flag[i_sample+1]:
                gravity_vector = acc_IMU[i_sample, :]
                gravity_vector_norm = norm(gravity_vector)
                euler_angles_esti[i_sample, 0] = np.arcsin(gravity_vector[1] / gravity_vector_norm)  # axis 0
                euler_angles_esti[i_sample, 1] = - np.arcsin(gravity_vector[0] / gravity_vector_norm)  # axis 1

                """2. Gyr integration"""
                for j_sample in range(i_sample, last_stance_end, -1):
                    euler_angles_esti[j_sample-1, :] = euler_angles_esti[j_sample, :] - angle_augments[j_sample, :]

                    """3. If j_sample is still stance phase"""
                    if stance_phase_flag[j_sample]:

                        acc_IMU_unified = acc_IMU[j_sample, :] / norm(acc_IMU[j_sample, :])
                        r = euler_angles_esti[j_sample, 0]
                        p = euler_angles_esti[j_sample, 1]
                        jacob = np.array([[0, -cos(p)],
                                          [cos(r) * cos(p), -sin(r) * sin(p)],
                                          [-sin(r) * cos(p), -cos(r) * sin(p)]])
                        f = np.array([[-sin(p) - acc_IMU_unified[0]],
                                      [sin(r) * cos(p) - acc_IMU_unified[1]],
                                      [cos(r) * cos(p) - acc_IMU_unified[2]]])
                        delta_f = np.matmul(jacob.T, f)
                        delta_f_normed = delta_f / norm(delta_f)
                        modi_coeff_dim_0 = p + np.arcsin(acc_IMU_unified[0])
                        if abs(acc_IMU_unified[1] / cos(p)) < 1:
                            modi_coeff_dim_1 = r - np.arcsin(acc_IMU_unified[1] / cos(p))
                        else:
                            modi_coeff_dim_1 = r - np.arcsin(acc_IMU_unified[1])
                        max_step = np.sqrt(modi_coeff_dim_0 ** 2 + modi_coeff_dim_1 ** 2)
                        if max_step > base_correction_coeff:
                            correction_coeff = base_correction_coeff
                        else:
                            correction_coeff = max_step
                        euler_angles_esti[j_sample, :2] = euler_angles_esti[j_sample, :2] - correction_coeff * delta_f_normed.T
                last_stance_end = i_sample
        return euler_angles_esti

    @staticmethod
    def get_euler_angles_gradient_decent(gait_data_df, stance_phase_flag, base_correction_coeff=0.01,
                                         cut_off_fre=None):
        delta_t = 1 / HAISHENG_SENSOR_SAMPLE_RATE

        acc_IMU = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values
        gyr_IMU = gait_data_df[['gyr_x', 'gyr_y', 'gyr_z']].values
        if cut_off_fre is not None:
            acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
            gyr_IMU = StrikeOffDetectorIMU.data_filt(gyr_IMU, cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
        data_len = gait_data_df.shape[0]

        angle_augments = gyr_IMU * delta_t
        euler_angles_esti = np.zeros([data_len, 3])

        init_start, euler_angles_esti = ProcessorFPA.find_first_stance(
            stance_phase_flag, gait_data_df[['acc_x', 'acc_y', 'acc_z']].values, euler_angles_esti)
        # initialize the following steps
        for i_sample in range(init_start + 1, data_len):
            euler_angles_esti[i_sample, :] = euler_angles_esti[i_sample - 1, :] + angle_augments[i_sample, :]
            if stance_phase_flag[i_sample]:
                acc_IMU_unified = acc_IMU[i_sample, :] / norm(acc_IMU[i_sample, :])

                r = euler_angles_esti[i_sample, 0]
                p = euler_angles_esti[i_sample, 1]
                jacob = np.array([[0, -cos(p)],
                                  [cos(r) * cos(p), -sin(r) * sin(p)],
                                  [-sin(r) * cos(p), -cos(r) * sin(p)]])
                f = np.array([[-sin(p) - acc_IMU_unified[0]],
                              [sin(r) * cos(p) - acc_IMU_unified[1]],
                              [cos(r) * cos(p) - acc_IMU_unified[2]]])

                delta_f = np.matmul(jacob.T, f)
                delta_f_normed = delta_f / norm(delta_f)

                modi_coeff_dim_0 = p + np.arcsin(acc_IMU_unified[0])
                if abs(acc_IMU_unified[1] / cos(p)) < 1:
                    modi_coeff_dim_1 = r - np.arcsin(acc_IMU_unified[1] / cos(p))
                else:
                    modi_coeff_dim_1 = r - np.arcsin(acc_IMU_unified[1])
                max_step = np.sqrt(modi_coeff_dim_0 ** 2 + modi_coeff_dim_1 ** 2)
                if max_step > base_correction_coeff:
                    correction_coeff = base_correction_coeff
                else:
                    correction_coeff = max_step
                euler_angles_esti[i_sample, :2] = euler_angles_esti[i_sample, :2] - correction_coeff * delta_f_normed.T

        return euler_angles_esti

    def get_rotated_acc(self, gait_data_df, euler_angles, acc_cut_off_fre=None):
        acc_IMU = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values
        if acc_cut_off_fre is not None:
            acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, acc_cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
        acc_IMU_rotated = np.zeros(acc_IMU.shape)
        data_len = acc_IMU.shape[0]
        for i_sample in range(data_len):
            dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
            acc_IMU_rotated[i_sample, :] = np.matmul(self._placement_R_foot_sensor, acc_IMU[i_sample, :].T)
            acc_IMU_rotated[i_sample, :] = np.matmul(dcm_mat, acc_IMU_rotated[i_sample, :].T)

        return acc_IMU_rotated

    @staticmethod
    def get_complementary_filtered_euler_angles(gait_data_df, stance_phase_flag, base_correction_coeff=0.2,
                                                cut_off_fre=None):
        delta_t = 1 / HAISHENG_SENSOR_SAMPLE_RATE

        acc_IMU = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values
        gyr_IMU = gait_data_df[['gyr_x', 'gyr_y', 'gyr_z']].values
        if cut_off_fre is not None:
            acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
            gyr_IMU = StrikeOffDetectorIMU.data_filt(gyr_IMU, cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
        data_len = gait_data_df.shape[0]

        # gyr_IMU_moved = np.zeros(gyr_IMU.shape)
        # gyr_IMU_moved[:-1, :] = gyr_IMU[1:, :]
        angle_augments = gyr_IMU * delta_t
        euler_angles_esti = np.zeros([data_len, 3])
        acc_IMU_norm = norm(acc_IMU, axis=1)

        # initialize orientation via first stance
        init_start, euler_angles_esti = ProcessorFPA.find_first_stance(
            stance_phase_flag, gait_data_df[['acc_x', 'acc_y', 'acc_z']].values, euler_angles_esti)

        # initialize the following steps
        for i_sample in range(init_start + 1, data_len):
            euler_angles_esti[i_sample, :] = euler_angles_esti[i_sample - 1, :] + angle_augments[i_sample, :]
            if stance_phase_flag[i_sample]:
                correction_coeff = base_correction_coeff
                pitch_correction = - np.arcsin(acc_IMU[i_sample, 0] / acc_IMU_norm[i_sample])
                roll_correction = np.arcsin(acc_IMU[i_sample, 1] / (acc_IMU_norm[i_sample] * cos(pitch_correction)))
                euler_angles_esti[i_sample, 0] = euler_angles_esti[i_sample, 0] + correction_coeff * \
                                                 (roll_correction - euler_angles_esti[i_sample, 0])
                euler_angles_esti[i_sample, 1] = euler_angles_esti[i_sample, 1] + correction_coeff * \
                                                 (pitch_correction - euler_angles_esti[i_sample, 1])
        return euler_angles_esti

    @staticmethod
    def initalize_steps_and_stance_phase(gait_param_df, stance_after_strike=15, stance_before_off=20):
        # stance_after_strike = 6, stance_before_off = 22
        stance_phase_sample_thd_lower = 0.3 * HAISHENG_SENSOR_SAMPLE_RATE
        stance_phase_sample_thd_higher = 1 * HAISHENG_SENSOR_SAMPLE_RATE
        # get stance phase
        strike_tuple = np.where(gait_param_df['strikes_IMU'] == 1)[0]
        off_tuple = np.where(gait_param_df['offs_IMU'] == 1)[0]
        data_len = gait_param_df.shape[0]
        strike_num = len(strike_tuple)
        steps = []
        stance_phase_flag = np.zeros([data_len], dtype=bool)
        abandoned_step_num = 0
        last_off = 0
        for i_strike in range(strike_num):
            strike = strike_tuple[i_strike]
            offs_near_strike = off_tuple[max(0, i_strike - 70): i_strike + 70]
            off = offs_near_strike[offs_near_strike > strike + stance_phase_sample_thd_lower]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:  # stance phase detected
                if strike < last_off:
                    continue
                off = off[0]
                steps.append([int(strike), int(off)])
                stance_phase_flag[strike + stance_after_strike:off - stance_before_off] = True
                last_off = off
            else:
                abandoned_step_num += 1
        return steps, stance_phase_flag

    def get_FPA_via_acc_before_strike_ratio(self, acc_IMU_rotated, steps, win_before_off, win_after_off):
        """Use the acc ratio before heel strike to calculate FPA"""
        data_len = acc_IMU_rotated.shape[0]
        FPA_estis = np.zeros([data_len])
        for step in steps:
            acc_clip = acc_IMU_rotated[step[0] - win_before_off:step[0] + win_after_off, 0:2]
            acc_sum = np.sum(acc_clip[35:55], axis=0)
            the_FPA_esti = np.arctan2(-acc_sum[0], -acc_sum[1]) * 180 / np.pi
            the_sample = int((step[0] + step[1]) / 2)
            FPA_estis[the_sample] = the_FPA_esti
        return FPA_estis
