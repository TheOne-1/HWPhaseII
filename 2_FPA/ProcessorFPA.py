from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation
from const import FILTER_WIN_LEN, MOCAP_SAMPLE_RATE, SUB_NAMES, PROCESSED_DATA_PATH
from StrikeOffDetectorIMU import StrikeOffDetectorIMU, StrikeOffDetectorIMUFilter
from numpy.linalg import norm
from transforms3d.euler import euler2mat


class ProcessorFPA(Processor):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=True, do_output_norm=False):
        super().__init__(train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm, do_output_norm)

        self.initialize_static_offset(1, 'l')
        self.angle_bias = [-7.167709058838527, -14.416528235055829, -11.931327513089638, -10.511167400927366,
                           -11.986275543322023, -13.935153882993816, -13.786796125522072, -10.466339744797462,
                           -14.202895204993775, -15.912388667389889, 0, 0, 0, 0, 0, -14.185046589390883, 0,
                           -13.196628072700516, -11.844226678542089]

    def convert_input_output(self, input_data, output_data, id_df, sampling_fre):
        if input_data is None:
            return None, None

        steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data)
        # convert input
        euler_angles_esti = self.get_kalman_filtered_euler_angles(input_data, id_df['trial_id'].values, stance_phase_flag)
        acc_IMU_rotated = self.get_rotated_acc(input_data, euler_angles_esti)
        FPA_estis, FPA_trues = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data)
        angle_bias = self.angle_bias[int(id_df['subject_id'][0])]
        FPA_estis = [fpa + angle_bias for fpa in FPA_estis]
        # FPA_estis = self.compensate_offset(FPA_estis, id_df)
        return np.array(FPA_estis), np.array(FPA_trues)

    # def compensate_offset(self, FPA_estis, id_df):
    #     sub_ids = id_df['subject_id'].values
    #     sub_id_list = list(set(sub_ids))
    #     for id in sub_id_list:
    #         id = int(id)
    #         FPA_estis[sub_ids == id] = FPA_estis[np.where(sub_ids == id)[0]] - self.angle_bias[id]
    #     return FPA_estis

    def white_box_solution(self):
        # the algorithm
        y_pred = self._x_test / 1.23 + 5

        # show results
        correlation_coeff, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred, precision=3)

        plt.figure()
        plt.plot(y_pred, self._y_test, '.')
        plt.xlabel('Predicted angles')
        plt.ylabel('True angles')
        plt.plot([-10, 50], [-10, 50], 'r-')
        plt.title('Correlation = ' + str(correlation_coeff) + '   mean error = ' + str(mean_error) +
                  '  RMSE = ' + str(RMSE[0]))

    def initalize_steps_and_stance_phase(self, input_data):
        stance_phase_sample_thd_lower = 0.5 * self.sensor_sampling_fre
        stance_phase_sample_thd_higher = 1 * self.sensor_sampling_fre
        # get stance phase
        strike_tuple = np.where(input_data[:, 6] == 1)[0]
        off_tuple = np.where(input_data[:, 7] == 1)[0]
        data_len = input_data.shape[0]
        strike_num = len(strike_tuple)
        steps = []
        stance_phase_flag = np.zeros([data_len], dtype=bool)
        for i_strike in range(strike_num):
            strike = strike_tuple[i_strike]
            offs_near_strike = off_tuple[max(0, i_strike - 10): i_strike + 10]
            off = offs_near_strike[offs_near_strike > strike + stance_phase_sample_thd_lower]
            off = off[off < strike + stance_phase_sample_thd_higher]
            if len(off) == 1:      # stance phase detected
                steps.append([int(strike), int(off)])
                stance_phase_flag[strike + 40:off[0] - 40] = True
        return steps, stance_phase_flag

    @staticmethod
    def get_rotated_acc(input_data, euler_angles):
        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMUFilter.data_filt(acc_IMU, 6, MOCAP_SAMPLE_RATE)
        acc_IMU_rotated = np.zeros(acc_IMU.shape)
        data_len = acc_IMU.shape[0]
        euler_angles = euler_angles
        for i_sample in range(data_len):
            dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
            acc_IMU_rotated[i_sample, :] = np.matmul(dcm_mat, acc_IMU[i_sample, :].T)
        return acc_IMU_rotated

    def get_FPA_via_max_acc_ratio(self, acc_IMU_rotated, steps, output_data):
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
                FPA_estis.append(the_FPA_esti)
        return FPA_estis, FPA_trues

    def get_kalman_filtered_euler_angles(self, input_data, trial_ids, stance_phase_flag, base_correction_coeff=0.03):
        delta_t = 1 / MOCAP_SAMPLE_RATE

        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMUFilter.data_filt(acc_IMU, 10, MOCAP_SAMPLE_RATE)
        gyr_IMU = input_data[:, 3:6]
        gyr_IMU = StrikeOffDetectorIMUFilter.data_filt(gyr_IMU, 10, MOCAP_SAMPLE_RATE)
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

    def initialize_static_offset(self, subject_id, side):
        subject_name = SUB_NAMES[subject_id]
        # get static trial data
        static_marker_file = PROCESSED_DATA_PATH + '\\' + subject_name + '\\' + '200Hz\\static.csv'
        data_static = pd.read_csv(static_marker_file)

        # get marker offset
        marker_column_names = [side.upper() + marker_name + '_' + axis_name for marker_name in ['FM2', 'FCC']
                               for axis_name in ['x', 'y']]
        data_static_marker = np.mean(data_static[marker_column_names])
        delta_x = -(data_static_marker[0] - data_static_marker[2])
        delta_y = data_static_marker[1] - data_static_marker[3]
        yaw_marker = np.rad2deg(np.arctan2(delta_x, delta_y))

        # get xsens orientation DCM
        yaw_xsens_raw = data_static[side + '_foot_yaw']
        yaw_xsens = np.mean(yaw_xsens_raw) + 180
        yaw_diff_degree = yaw_xsens - yaw_marker
        return yaw_diff_degree





