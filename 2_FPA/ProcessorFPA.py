from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation
from const import FILTER_WIN_LEN, MOCAP_SAMPLE_RATE, SUB_NAMES
from StrikeOffDetectorIMU import StrikeOffDetectorIMUFilter
from numpy.linalg import norm
from transforms3d.euler import euler2mat
from sklearn.linear_model import LinearRegression


class ProcessorFPA(Processor):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=True, do_output_norm=False):
        super().__init__(train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm, do_output_norm)

        # 0 for normal prediction, 1 for best empirical equation parameter via linear regression,
        # 2 for best cut-off frequency
        self.get_best_param = 0

        self.angle_bias = [5.6288632361159046, 11.452744870987829, 10.68706348139574, 10.994062733806373,
                           8.358435315266064, 11.69003309005015, 12.636716007510103, 10.650004666349162,
                           8.827560704018282, 12.832568360830777, 15.543915822029902, 9.262269550019823,
                           9.823769726131474, 0, 0, 0, 0, 0, 0]

    def convert_input_output(self, input_data, output_data, id_df, sampling_fre):
        if input_data is None:
            return None, None

        if not self.get_best_param:
            sub_ids = id_df['subject_id'].values
            sub_id_list = list(set(sub_ids))
            predict_result_df = pd.DataFrame()

            for sub_id in sub_id_list:
                sub_id = int(sub_id)
                sub_data_index = id_df['subject_id'] == sub_id
                input_data_sub = input_data[sub_data_index]
                output_data_sub = output_data[sub_data_index]
                steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data_sub)
                # convert input
                euler_angles_esti = self.get_kalman_filtered_euler_angles(input_data_sub, id_df['trial_id'].values,
                                                                          stance_phase_flag)
                acc_IMU_rotated = self.get_rotated_acc(input_data_sub, euler_angles_esti)
                angle_bias = self.angle_bias[sub_id]
                FPA_estis, FPA_trues = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data_sub, angle_bias)

                pearson_coeff, RMSE, mean_error = Evaluation.plot_nn_result(FPA_trues, FPA_estis, title=SUB_NAMES[sub_id])
                predict_result_df = Evaluation.insert_prediction_result(
                    predict_result_df, SUB_NAMES[sub_id], pearson_coeff, RMSE, mean_error)
            Evaluation.export_prediction_result(predict_result_df)

        elif self.get_best_param == 1:
            # Use linear regression to get the best empirical equation parameter
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data)
            euler_angles_esti = self.get_kalman_filtered_euler_angles(input_data, id_df['trial_id'].values,
                                                                      stance_phase_flag)
            acc_IMU_rotated = self.get_rotated_acc(input_data, euler_angles_esti)
            FPA_estis, FPA_trues = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data, 0, False)
            model = LinearRegression()
            model.fit(FPA_estis.reshape(-1, 1), FPA_trues)
            print('a = ' + str(model.coef_[0]) + '   b = ' + str(model.intercept_))

        elif self.get_best_param == 2:
            # find the best filter cut-off frequency
            predict_result_df = pd.DataFrame()
            for stance_end in np.arange(30, 51, 2):
                steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data, stance_end)
                euler_angles_esti = self.get_kalman_filtered_euler_angles(
                    input_data, id_df['trial_id'].values, stance_phase_flag, base_correction_coeff=0.065, cut_off_fre=12)
                acc_IMU_rotated = self.get_rotated_acc(input_data, euler_angles_esti, acc_cut_off_fre=4)
                FPA_estis, FPA_trues = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data, 0)
                pearson_coeff, RMSE, mean_error = Evaluation.plot_nn_result(FPA_trues, FPA_estis, title='All')
                predict_result_df = Evaluation.insert_prediction_result(
                    predict_result_df, stance_end, pearson_coeff, RMSE, mean_error)
            Evaluation.export_prediction_result(predict_result_df)
        return None, None

    def white_box_solution(self):
        # the algorithm
        pass

    def initalize_steps_and_stance_phase(self, input_data):
        stance_phase_sample_thd_lower = 0.3 * self.sensor_sampling_fre
        stance_phase_sample_thd_higher = 1 * self.sensor_sampling_fre
        # get stance phase
        strike_tuple = np.where(input_data[:, 6] == 1)[0]
        off_tuple = np.where(input_data[:, 7] == 1)[0]
        data_len = input_data.shape[0]
        strike_num = len(strike_tuple)
        steps = []
        stance_phase_flag = np.zeros([data_len], dtype=bool)
        abandoned_step_num = 0
        for i_strike in range(strike_num):
            strike = strike_tuple[i_strike]
            offs_near_strike = off_tuple[max(0, i_strike - 70): i_strike + 70]
            off = offs_near_strike[offs_near_strike > strike + stance_phase_sample_thd_lower]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:      # stance phase detected
                steps.append([int(strike), int(off)])
                stance_phase_flag[strike + 31:off[0] - 40] = True
            else:
                abandoned_step_num += 1
        print('{num} steps abandoned'.format(num=abandoned_step_num))
        return steps, stance_phase_flag

    @staticmethod
    def get_rotated_acc(input_data, euler_angles, acc_cut_off_fre=4):
        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMUFilter.data_filt(acc_IMU, acc_cut_off_fre, MOCAP_SAMPLE_RATE)
        acc_IMU_rotated = np.zeros(acc_IMU.shape)
        data_len = acc_IMU.shape[0]
        euler_angles = euler_angles
        for i_sample in range(data_len):
            dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
            acc_IMU_rotated[i_sample, :] = np.matmul(dcm_mat, acc_IMU[i_sample, :].T)
        return acc_IMU_rotated

    def get_FPA_via_max_acc_ratio(self, acc_IMU_rotated, steps, output_data, angle_bias=0, use_empirical=True):
        filter_delay = int(FILTER_WIN_LEN / 2)
        win_before_off = int(0.08 * self.sensor_sampling_fre)
        win_after_off = int(0.12 * self.sensor_sampling_fre)
        FPA_estis, FPA_trues = [], []
        for step in steps:
            the_FPA_true = np.max(output_data[step[0] - filter_delay:step[1] - filter_delay])
            if the_FPA_true:
                FPA_trues.append(the_FPA_true)

                max_acc_x = np.max(acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 0])
                max_acc_y = np.max(acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 1])

                # max_acc_x_index = np.argmax(acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 0])
                # max_acc_x_start = step[1]-win_before_off+max_acc_x_index
                # max_acc_x = np.mean(acc_IMU_rotated[max_acc_x_start-2:max_acc_x_start+3, 0])
                # max_acc_y_index = np.argmax(acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 1])
                # max_acc_y_start = step[1]-win_before_off+max_acc_y_index
                # max_acc_y = np.mean(acc_IMU_rotated[max_acc_y_start-2:max_acc_y_start+3, 1])

                the_FPA_esti = np.arctan2(max_acc_x, max_acc_y) * 180 / np.pi
                # the_FPA_esti = the_FPA_esti + angle_bias        # add the bias
                if use_empirical:
                    the_FPA_esti = the_FPA_esti * 0.938 - 1.92        # the empirical function
                FPA_estis.append(the_FPA_esti)
        return np.array(FPA_estis), np.array(FPA_trues)

    def get_kalman_filtered_euler_angles(self, input_data, trial_ids, stance_phase_flag, base_correction_coeff=0.065,
                                         cut_off_fre=12):
        delta_t = 1 / MOCAP_SAMPLE_RATE

        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMUFilter.data_filt(acc_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        gyr_IMU = input_data[:, 3:6]
        gyr_IMU = StrikeOffDetectorIMUFilter.data_filt(gyr_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        data_len = input_data.shape[0]

        angle_augments = gyr_IMU * delta_t
        euler_angles_esti = np.zeros([data_len, 3])
        acc_IMU_norm = norm(acc_IMU, axis=1)
        roll_correction = np.arctan2(acc_IMU[:, 1], acc_IMU_norm)          # axis 0
        pitch_correction = - np.arctan2(acc_IMU[:, 0], acc_IMU_norm)       # axis 1

        dynamic_correction_coeff = 0.9
        for i_sample in range(data_len):
            if trial_ids[i_sample] != trial_ids[i_sample-1]:
                dynamic_correction_coeff = 0.9
            euler_angles_esti[i_sample, :] = euler_angles_esti[i_sample - 1, :] + angle_augments[i_sample, :]

            if stance_phase_flag[i_sample]:
                if dynamic_correction_coeff > 1e-3:
                    correction_coeff = dynamic_correction_coeff + base_correction_coeff
                    dynamic_correction_coeff = dynamic_correction_coeff * 0.9
                else:
                    correction_coeff = base_correction_coeff
                euler_angles_esti[i_sample, 0] = euler_angles_esti[i_sample, 0] + correction_coeff * \
                    (roll_correction[i_sample] - euler_angles_esti[i_sample, 0])
                euler_angles_esti[i_sample, 1] = euler_angles_esti[i_sample, 1] + correction_coeff * \
                    (pitch_correction[i_sample] - euler_angles_esti[i_sample, 1])
        return euler_angles_esti

