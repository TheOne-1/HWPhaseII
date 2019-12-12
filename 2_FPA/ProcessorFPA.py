from Processor import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation
from const import FILTER_WIN_LEN, MOCAP_SAMPLE_RATE, SUB_NAMES, TRIAL_NAMES
from StrikeOffDetectorIMU import StrikeOffDetectorIMU
from numpy.linalg import norm
from numpy import sin, cos
from transforms3d.euler import euler2mat
from sklearn.linear_model import LinearRegression


class ProcessorFPA(Processor):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm=True, do_output_norm=False):
        super().__init__(train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, grf_side, param_name,
                 IMU_location, data_type, do_input_norm, do_output_norm)

        # 0 for normal prediction, 1 for best empirical equation parameter via linear regression,
        # 2 for best cut-off frequency
        self.experiment_id = 4

    def convert_input_output(self, input_data, output_data, supp_df, sampling_fre):
        if input_data is None:
            return None, None

        if self.experiment_id == 4:
            # brand new functions for publication
            sub_id_list = np.sort(list(set(supp_df['subject_id'].values))).astype('int')
            trial_id_list = np.sort(list(set(supp_df['trial_id'].values))).astype('int')
            for sub_id in sub_id_list:
                sub_id = int(sub_id)
                for trial_id in trial_id_list:
                    data_index = (supp_df['subject_id'] == sub_id) & (supp_df['trial_id'] == trial_id)
                    input_data_sub = input_data[data_index]
                    output_data_sub = output_data[data_index]
                    steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data_sub)
                    euler_angles_true = supp_df[['l_foot_roll', 'l_foot_pitch', 'l_foot_yaw']].values

                    # euler_angles_esti = self.get_complementary_filtered_euler_angles_new(input_data_sub,
                    #                                                           stance_phase_flag)
                    # euler_angles_esti = self.get_complementary_filtered_euler_angles(input_data, supp_df['trial_id'].values,
                    #                                                           stance_phase_flag)
                    euler_angles_esti = self.get_euler_angles_gradient_decent(input_data_sub,
                                                                              stance_phase_flag)

                    euler_angles_esti[:, 0] = euler_angles_esti[:, 0]
                    euler_angles_true = np.deg2rad(euler_angles_true)
                    plt.figure()
                    plt.plot(euler_angles_esti[:, 0])
                    plt.plot(euler_angles_true[:, 0])
                    plt.figure()
                    plt.plot(euler_angles_esti[:, 1])
                    plt.plot(euler_angles_true[:, 1])

                    acc_IMU_rotated = self.get_rotated_acc(input_data_sub, euler_angles_esti)
                    FPA_estis, FPA_trues, _ = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data_sub)
                    pearson_coeff, RMSE, mean_error = Evaluation.plot_fpa_result(FPA_trues, FPA_estis, sub_id)
                    plt.show()
                    # ProcessorFPA.save_trial_result(sub_id, trial_id, FPA_estis, steps)


        elif self.experiment_id == 0:
            sub_ids = supp_df['subject_id'].values
            sub_id_list = list(set(sub_ids))
            predict_result_df = pd.DataFrame()

            for sub_id in sub_id_list:
                sub_id = int(sub_id)
                data_index = supp_df['subject_id'] == sub_id
                input_data_sub = input_data[data_index]
                output_data_sub = output_data[data_index]
                steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data_sub)
                # convert input
                euler_angles_esti = self.get_complementary_filtered_euler_angles(input_data_sub, supp_df['trial_id'].values,
                                                                                 stance_phase_flag)
                acc_IMU_rotated = self.get_rotated_acc(input_data_sub, euler_angles_esti)
                FPA_estis, FPA_trues, _ = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data_sub)

                pearson_coeff, RMSE, mean_error = Evaluation.plot_fpa_result(FPA_trues, FPA_estis, sub_id)
                predict_result_df = Evaluation.insert_prediction_result(
                    predict_result_df, SUB_NAMES[sub_id], pearson_coeff, RMSE, mean_error)
            Evaluation.export_prediction_result(predict_result_df)

        elif self.experiment_id == 1:
            # Use linear regression to get the best empirical equation parameter
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data)
            euler_angles_esti = self.get_complementary_filtered_euler_angles(input_data, supp_df['trial_id'].values,
                                                                             stance_phase_flag)
            acc_IMU_rotated = self.get_rotated_acc(input_data, euler_angles_esti)
            FPA_estis, FPA_trues, _ = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data, False)
            model = LinearRegression()
            model.fit(FPA_estis.reshape(-1, 1), FPA_trues)
            print('a = ' + str(model.coef_[0]) + '   b = ' + str(model.intercept_))

        elif self.experiment_id == 2:
            # find the best filter cut-off frequency
            predict_result_df = pd.DataFrame()
            for stance_end in np.arange(30, 51, 2):
                steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data)
                euler_angles_esti = self.get_complementary_filtered_euler_angles(
                    input_data, supp_df['trial_id'].values, stance_phase_flag, base_correction_coeff=0.065, cut_off_fre=12)
                acc_IMU_rotated = self.get_rotated_acc(input_data, euler_angles_esti, acc_cut_off_fre=4)
                FPA_estis, FPA_trues, _ = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data)
                pearson_coeff, RMSE, mean_error = Evaluation.plot_fpa_result(FPA_trues, FPA_estis, -1)
                predict_result_df = Evaluation.insert_prediction_result(
                    predict_result_df, stance_end, pearson_coeff, RMSE, mean_error)
            Evaluation.export_prediction_result(predict_result_df)

        elif self.experiment_id == 3:
            # record the estimation result of each step
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(input_data)
            # convert input
            euler_angles_esti = self.get_complementary_filtered_euler_angles(input_data, supp_df['trial_id'].values,
                                                                             stance_phase_flag)
            acc_IMU_rotated = self.get_rotated_acc(input_data, euler_angles_esti)
            FPA_estis, FPA_trues, steps_used = self.get_FPA_via_max_acc_ratio(acc_IMU_rotated, steps, output_data)
            detailed_result_df = self.get_detailed_result_df(supp_df, FPA_estis, FPA_trues, steps_used)
            detailed_result_df.to_csv('detailed_result_df.csv', index=False)
        return None, None

    @staticmethod
    def save_trial_result(sub_id, trial_id, FPA_estis, steps):
        step_flags = np.zeros(FPA_estis.shape)
        for step in steps:
            step_flags[step[0]] = 1     # strike
            step_flags[step[1]] = 2     # off
        sub_name = SUB_NAMES[sub_id]
        trial_name = TRIAL_NAMES[trial_id]
        file_path = 'step_result/' + sub_name + '/step_result_of_' + trial_name + '.csv'
        tbme_data = pd.read_csv(file_path, index_col=False)
        tbme_data['fpa_acc_ratio'] = FPA_estis
        tbme_data['step_flag'] = step_flags
        tbme_data.to_csv(file_path, index=False)

    @staticmethod
    def get_detailed_result_df(id_df, FPA_estis, FPA_trues, steps_used):
        id_df = id_df.astype(int)
        data_len = len(FPA_estis)
        id_df_row_index = []
        for i_step in range(data_len):
            id_df_row_index.append(steps_used[i_step][1])
        detailed_result_df = id_df.iloc[id_df_row_index, :]
        detailed_result_df.insert(loc=0, column='FPA true', value=FPA_trues)
        detailed_result_df.insert(loc=0, column='FPA esti', value=FPA_estis)
        detailed_result_df = detailed_result_df.reset_index(drop=True)
        return detailed_result_df

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
                off = off[0]
                steps.append([int(strike), int(off)])
                stance_phase_flag[strike + 11:off - 20] = True
            else:
                abandoned_step_num += 1
        print('{num} steps abandoned'.format(num=abandoned_step_num))
        return steps, stance_phase_flag

    @staticmethod
    def get_rotated_acc(input_data, euler_angles, acc_cut_off_fre=2):
        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, acc_cut_off_fre, MOCAP_SAMPLE_RATE)
        acc_IMU_rotated = np.zeros(acc_IMU.shape)
        data_len = acc_IMU.shape[0]
        for i_sample in range(data_len):
            dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
            acc_IMU_rotated[i_sample, :] = np.matmul(dcm_mat, acc_IMU[i_sample, :].T)
        return acc_IMU_rotated

    def get_FPA_via_max_acc_ratio(self, acc_IMU_rotated, steps, output_data, use_empirical=True):

        filter_delay = 0
        win_before_off = int(0.08 * self.sensor_sampling_fre)
        win_after_off = int(0.12 * self.sensor_sampling_fre)
        FPA_estis, FPA_trues, steps_used = [], [], []
        for step in steps:
            # get true FPA values
            output_clip = output_data[step[0] - filter_delay:step[1] - filter_delay]
            output_clip = output_clip[output_clip != 0]
            if len(output_clip) != 1:
                print('multiple true FPA found, step skipped')
                continue
            the_FPA_true = output_clip[0]
            if the_FPA_true:
                FPA_trues.append(the_FPA_true)
                steps_used.append(step)

                acc_x_clip = acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 0]
                acc_y_clip = acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 1]
                max_acc_x = np.max(acc_x_clip)
                max_acc_y = np.max(acc_y_clip)

                if max_acc_x < 1:
                    max_acc_y_arg = np.argmax(acc_y_clip)
                    max_acc_x = acc_x_clip[max_acc_y_arg - 5]
                the_FPA_esti = np.arctan2(max_acc_x, max_acc_y) * 180 / np.pi
                if use_empirical:
                    the_FPA_esti = the_FPA_esti - 4  # the empirical function

                FPA_estis.append(the_FPA_esti)
        return np.array(FPA_estis), np.array(FPA_trues), steps_used

    def get_complementary_filtered_euler_angles(self, input_data, trial_ids, stance_phase_flag, base_correction_coeff=0.065,
                                                cut_off_fre=6):
        delta_t = 1 / MOCAP_SAMPLE_RATE

        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        gyr_IMU = input_data[:, 3:6]
        gyr_IMU = StrikeOffDetectorIMU.data_filt(gyr_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        data_len = input_data.shape[0]

        gyr_IMU_moved = np.zeros(gyr_IMU.shape)
        gyr_IMU_moved[:-1, :] = gyr_IMU[1:, :]
        angle_augments = (gyr_IMU + gyr_IMU_moved) / 2 * delta_t
        # angle_augments = gyr_IMU * delta_t
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

    def get_FPA_via_max_acc_ratio_new(self, acc_IMU_rotated, steps, output_data, use_empirical=True):
        win_before_off = int(0.08 * self.sensor_sampling_fre)
        win_after_off = int(0.12 * self.sensor_sampling_fre)
        data_len = acc_IMU_rotated.shape[0]
        FPA_estis = np.zeros([data_len])
        for step in steps:
            acc_x_clip = acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 0]
            acc_y_clip = acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 1]
            max_acc_y, max_acc_y_index = np.max(acc_y_clip), np.argmax(acc_y_clip)
            max_acc_x = acc_x_clip[max_acc_y_index]

            the_FPA_esti = np.arctan2(max_acc_x, max_acc_y) * 180 / np.pi
            the_sample = int((step[0] + step[1]) / 2)
            FPA_estis[the_sample] = the_FPA_esti
        return np.array(FPA_estis)

    @staticmethod
    def get_complementary_filtered_euler_angles_new(input_data, stance_phase_flag, base_correction_coeff=0.065,
                                                    cut_off_fre=6):
        delta_t = 1 / MOCAP_SAMPLE_RATE

        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        gyr_IMU = input_data[:, 3:6]
        gyr_IMU = StrikeOffDetectorIMU.data_filt(gyr_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        data_len = input_data.shape[0]

        gyr_IMU_moved = np.zeros(gyr_IMU.shape)
        gyr_IMU_moved[:-1, :] = gyr_IMU[1:, :]
        angle_augments = (gyr_IMU + gyr_IMU_moved) / 2 * delta_t
        # angle_augments = gyr_IMU * delta_t
        euler_angles_esti = np.zeros([data_len, 3])
        acc_IMU_norm = norm(acc_IMU, axis=1)

        # initialize orientation via first stance
        init_start, euler_angles_esti = ProcessorFPA.find_first_stance(stance_phase_flag, input_data, euler_angles_esti)

        # initialize the following steps
        for i_sample in range(init_start+1, data_len):
            euler_angles_esti[i_sample, :] = euler_angles_esti[i_sample - 1, :] + angle_augments[i_sample, :]
            if stance_phase_flag[i_sample]:
                correction_coeff = base_correction_coeff
                pitch_correction = - np.arcsin(acc_IMU[i_sample, 0]/acc_IMU_norm[i_sample])
                roll_correction = np.arcsin(acc_IMU[i_sample, 1]/(acc_IMU_norm[i_sample]*cos(pitch_correction)))
                euler_angles_esti[i_sample, 0] = euler_angles_esti[i_sample, 0] + correction_coeff * \
                    (roll_correction - euler_angles_esti[i_sample, 0])
                euler_angles_esti[i_sample, 1] = euler_angles_esti[i_sample, 1] + correction_coeff * \
                    (pitch_correction - euler_angles_esti[i_sample, 1])
        return euler_angles_esti

    @staticmethod
    def get_euler_angles_gradient_decent(input_data, stance_phase_flag, base_correction_coeff=0.005,
                                         cut_off_fre=6):
        delta_t = 1 / MOCAP_SAMPLE_RATE

        acc_IMU = input_data[:, 0:3]
        acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        gyr_IMU = input_data[:, 3:6]
        gyr_IMU = StrikeOffDetectorIMU.data_filt(gyr_IMU, cut_off_fre, MOCAP_SAMPLE_RATE)
        data_len = input_data.shape[0]

        gyr_IMU_moved = np.zeros(gyr_IMU.shape)
        gyr_IMU_moved[:-1, :] = gyr_IMU[1:, :]
        angle_augments = (gyr_IMU + gyr_IMU_moved) / 2 * delta_t
        # angle_augments = gyr_IMU * delta_t
        euler_angles_esti = np.zeros([data_len, 3])

        init_start, euler_angles_esti = ProcessorFPA.find_first_stance(stance_phase_flag, input_data, euler_angles_esti)
        # initialize the following steps
        gravity = 9.81
        for i_sample in range(init_start+1, data_len):
            euler_angles_esti[i_sample, :] = euler_angles_esti[i_sample - 1, :] + angle_augments[i_sample, :]
            if stance_phase_flag[i_sample]:
                acc_IMU_unified = acc_IMU[i_sample, :] / norm(acc_IMU[i_sample, :])

                r = euler_angles_esti[i_sample, 0]
                p = euler_angles_esti[i_sample, 1]
                jacob = np.array([[0,               -cos(p)],
                                  [cos(r) * cos(p), -sin(r) * sin(p)],
                                  [-sin(r)*cos(p),  -cos(r)*sin(p)]])
                f = np.array([[-sin(p) - acc_IMU_unified[0]],
                              sin(r) * cos(p) - [acc_IMU_unified[1]],
                              cos(r) * cos(p) - [acc_IMU_unified[2]]])

                delta_f = np.matmul(jacob.T, f)
                delta_f_normed = delta_f / norm(delta_f)

                max_step = np.sqrt((p + np.arcsin(acc_IMU_unified[0]))**2 + (r - np.arcsin(acc_IMU_unified[1]/cos(p)))**2)
                if max_step > base_correction_coeff:
                    correction_coeff = 0.005
                else:
                    correction_coeff = max_step
                euler_angles_esti[i_sample, :2] = euler_angles_esti[i_sample, :2] - correction_coeff * delta_f_normed.T

        return euler_angles_esti

    @staticmethod
    def find_first_stance(stance_phase_flag, input_data, euler_angles_esti):
        # initialize orientation via first stance
        ini_start_delay, ini_len = 40, 10
        i_sample = 0

        while not stance_phase_flag[i_sample]:
            i_sample += 1
        init_start = i_sample + ini_start_delay
        init_end = init_start + ini_len
        init_data_clip = input_data[init_start:init_end, :]
        gravity_vector = np.mean(init_data_clip[:, :3], axis=0)
        gravity_vector_norm = norm(gravity_vector)
        euler_angles_esti[init_start, 0] = np.arcsin(gravity_vector[1]/gravity_vector_norm)          # axis 0
        euler_angles_esti[init_start, 1] = - np.arcsin(gravity_vector[0]/gravity_vector_norm)        # axis 1
        return init_start, euler_angles_esti

