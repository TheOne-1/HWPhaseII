import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from const import DATA_PATH_HS, SUB_AND_TRIALS_HS, DATA_COLUMNS_IMU, MARKERS_HS, SUB_SELECTED_SPEEDS, \
    TRIAL_NAMES_HS, HAISHENG_SENSOR_SAMPLE_RATE, COLUMN_NAMES_HAISHENG
from numpy.linalg import norm
from StrikeOffDetectorIMU import StrikeOffDetectorIMU
from numpy import cos, sin
from ProcessorFPA import ProcessorFPA
from transforms3d.euler import euler2mat
from DataProcessorHS import DataInitializerHS


class InitFPA:
    def __init__(self, sub_folder):
        self.sub_folder = sub_folder
        self.sub_name = sub_folder.split('_')[-1]
        self._placement_offset = DataInitializerHS.init_placement_offset(sub_folder)

    def start_init(self):
        base_path = DATA_PATH_HS + 'processed/' + self.sub_folder + '/'
        fpa_true_list, fpa_esti_list = [], []
        for trial_id in range(len(TRIAL_NAMES_HS)):
            print(TRIAL_NAMES_HS[trial_id])
            gait_data_path = base_path + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_data_df = pd.read_csv(gait_data_path, index_col=False)

            gait_param_path = base_path + 'param_of_' + TRIAL_NAMES_HS[trial_id] + '.csv'
            gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            output_data = gait_param_df['FPA_true'].values.reshape(-1, 1)
            steps, stance_phase_flag = self.initalize_steps_and_stance_phase(gait_param_df)

            # plt.figure()
            # grf = -gait_data_df['f_1_z']
            # plt.plot(grf, 'r-')
            # plt.plot(stance_phase_flag*20, 'green')
            # plt.show()

            euler_angles_esti = self.get_complementary_filtered_euler_angles(
                gait_data_df, stance_phase_flag, cut_off_fre=6)
            # acc_IMU_rotated = self.get_rotated_acc_new_filter(gait_data_df, euler_angles_esti)
            acc_IMU_rotated = self.get_rotated_acc(gait_data_df, euler_angles_esti, acc_cut_off_fre=6)
            FPA_estis = self.get_FPA_via_max_acc_ratio_at_norm_peak(acc_IMU_rotated, steps)
            # FPA_estis = gait_param_df['FPA_tbme'].values
            fpa_true_temp, fpa_esti_temp = ProcessorFPA.compare_result(FPA_estis, output_data, steps)
            fpa_true_list.extend(fpa_true_temp[20:-20])
            fpa_esti_list.extend(fpa_esti_temp[20:-20])
            fpa_esti_list = [esti - self._placement_offset for esti in fpa_esti_list]

            # plt.figure()
            plt.plot(fpa_true_temp, fpa_esti_temp, '.')
        plt.plot([-20, 60], [-20, 60], 'black')
        plt.title('FPA via new acc ratio')

    @staticmethod
    def get_FPA_via_max_acc_ratio_at_norm_peak(acc_IMU_rotated, steps):
        win_before_off = int(0.07 * HAISHENG_SENSOR_SAMPLE_RATE)
        win_after_off = int(0.13 * HAISHENG_SENSOR_SAMPLE_RATE)
        data_len = acc_IMU_rotated.shape[0]
        FPA_estis = np.zeros([data_len])
        planar_acc_norm = norm(acc_IMU_rotated[:, :2], axis=1)
        for step in steps:
            acc_x_clip = acc_IMU_rotated[step[1] - win_before_off:step[1] + win_after_off, 0]
            acc_y_clip = acc_IMU_rotated[step[1] - win_before_off:step[1] + win_after_off, 1]

            # max_acc_x = np.max(acc_x_clip)
            # max_acc_y = np.max(acc_y_clip)
            # the_FPA_esti = np.arctan2(max_acc_x, max_acc_y) * 180 / np.pi

            acc_norm_clip = planar_acc_norm[step[1] - win_before_off:step[1] + win_after_off]
            max_acc_norm_index = np.argmax(acc_norm_clip)
            the_FPA_esti = np.arctan2(acc_x_clip[max_acc_norm_index], acc_y_clip[max_acc_norm_index]) * 180 / np.pi

            the_sample = int((step[0] + step[1]) / 2)
            FPA_estis[the_sample] = the_FPA_esti

            # plt.figure()
            # plt.plot(acc_x_clip)
            # plt.plot(acc_y_clip)
            # plt.show()
            # if 4600 < step[1] < 4700:
            #     plt.figure()
            #     plt.plot(acc_x_clip)
            #     plt.plot(acc_y_clip)
            #     plt.plot(acc_norm_clip)
            #     # plt.show()
                # pass
        return FPA_estis

    @staticmethod
    def get_rotated_acc(gait_data_df, euler_angles, acc_cut_off_fre=None):
        acc_IMU = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values
        if acc_cut_off_fre is not None:
            acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, acc_cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
        acc_IMU_rotated = np.zeros(acc_IMU.shape)
        data_len = acc_IMU.shape[0]
        for i_sample in range(data_len):
            dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
            acc_IMU_rotated[i_sample, :] = np.matmul(dcm_mat, acc_IMU[i_sample, :].T)
        return acc_IMU_rotated

    @staticmethod
    def get_rotated_acc_new_filter(gait_data_df, euler_angles):
        acc_IMU_unfilted = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values

        acc_IMU = np.zeros(acc_IMU_unfilted.shape)
        for i_axis in range(3):
            acc_IMU[:, i_axis] = ProcessorFPA.smooth(acc_IMU_unfilted[:, i_axis], 50)
            # plt.figure()
            # plt.plot(acc_IMU[:, i_axis])
            # plt.plot(acc_IMU_unfilted[:, i_axis])

        # acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU_unfilted, 2, 100)
        # for i_axis in range(3):
        #     plt.figure()
        #     plt.plot(acc_IMU[:, i_axis])
        #     plt.plot(acc_IMU_unfilted[:, i_axis])

        acc_IMU_rotated = np.zeros(acc_IMU.shape)
        data_len = acc_IMU.shape[0]
        for i_sample in range(data_len):
            dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
            acc_IMU_rotated[i_sample, :] = np.matmul(dcm_mat, acc_IMU[i_sample, :].T)
        return acc_IMU_rotated

    @staticmethod
    def get_complementary_filtered_euler_angles(gait_data_df, stance_phase_flag, base_correction_coeff=0.2,
                                                cut_off_fre=6):
        delta_t = 1 / HAISHENG_SENSOR_SAMPLE_RATE

        acc_IMU = gait_data_df[['acc_x', 'acc_y', 'acc_z']].values
        acc_IMU = StrikeOffDetectorIMU.data_filt(acc_IMU, cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
        gyr_IMU = gait_data_df[['gyr_x', 'gyr_y', 'gyr_z']].values
        gyr_IMU = StrikeOffDetectorIMU.data_filt(gyr_IMU, cut_off_fre, HAISHENG_SENSOR_SAMPLE_RATE)
        data_len = gait_data_df.shape[0]

        gyr_IMU_moved = np.zeros(gyr_IMU.shape)
        gyr_IMU_moved[:-1, :] = gyr_IMU[1:, :]
        angle_augments = (gyr_IMU + gyr_IMU_moved) / 2 * delta_t
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
    def initalize_steps_and_stance_phase(gait_param_df, stance_after_strike=19, stance_before_off=19):
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

    def get_FPA_via_displacement_ratio(self, acc_IMU_rotated, steps, win_before_off=30, win_after_off=50):
        """Do integration till speed peak"""
        data_len = acc_IMU_rotated.shape[0]
        FPA_estis = np.zeros([data_len])
        delta_t = 1 / HAISHENG_SENSOR_SAMPLE_RATE
        for step in steps:
            acc_x_clip = acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 0]
            acc_y_clip = acc_IMU_rotated[step[1]-win_before_off:step[1]+win_after_off, 1]
            speed_x, speed_y = np.zeros(acc_x_clip.shape), np.zeros(acc_y_clip.shape)
            disp_x, disp_y = np.zeros(acc_x_clip.shape), np.zeros(acc_y_clip.shape)
            for i_sample in range(acc_x_clip.shape[0]):
                speed_x[i_sample] = speed_x[i_sample-1] + acc_x_clip[i_sample] * delta_t
                speed_y[i_sample] = speed_y[i_sample-1] + acc_y_clip[i_sample] * delta_t
                disp_x[i_sample] = disp_x[i_sample-1] + delta_t * (speed_x[i_sample] + speed_x[i_sample-1]) / 2
                disp_y[i_sample] = disp_y[i_sample-1] + delta_t * (speed_y[i_sample] + speed_y[i_sample-1]) / 2
            speed_peak_y_index = np.argmax(speed_y)

            # plt.figure()
            # plt.plot(disp_x[:speed_peak_y_index])
            # plt.plot(disp_y[:speed_peak_y_index])
            # plt.show()

            the_FPA_esti = np.arctan2(disp_x[speed_peak_y_index], disp_y[speed_peak_y_index]) * 180 / np.pi
            the_sample = int((step[0] + step[1]) / 2)
            FPA_estis[the_sample] = the_FPA_esti
        return FPA_estis

    def get_FPA_via_displacement_ratio_new(self, acc_IMU_rotated, steps, win_before_off=60, win_after_off=100):
        """Do integration of the whole next step"""
        data_len = acc_IMU_rotated.shape[0]
        FPA_estis = np.zeros([data_len])
        delta_t = 1 / HAISHENG_SENSOR_SAMPLE_RATE
        for i_step in range(1, len(steps)-1):
            the_last_step = steps[i_step - 1]
            step = steps[i_step]
            the_next_step = steps[i_step + 1]
            acc_x_clip = np.concatenate([acc_IMU_rotated[the_last_step[1]-win_before_off:step[0], 0], acc_IMU_rotated[step[1]-win_before_off:the_next_step[0], 0]])
            acc_y_clip = np.concatenate([acc_IMU_rotated[the_last_step[1]-win_before_off:step[0], 1], acc_IMU_rotated[step[1]-win_before_off:the_next_step[0], 1]])
            speed_x, speed_y = np.zeros(acc_x_clip.shape), np.zeros(acc_y_clip.shape)
            disp_x, disp_y = np.zeros(acc_x_clip.shape), np.zeros(acc_y_clip.shape)
            for i_sample in range(acc_x_clip.shape[0]):
                speed_x[i_sample] = speed_x[i_sample-1] + acc_x_clip[i_sample] * delta_t
                speed_y[i_sample] = speed_y[i_sample-1] + acc_y_clip[i_sample] * delta_t
                disp_x[i_sample] = disp_x[i_sample-1] + delta_t * (speed_x[i_sample] + speed_x[i_sample-1]) / 2
                disp_y[i_sample] = disp_y[i_sample-1] + delta_t * (speed_y[i_sample] + speed_y[i_sample-1]) / 2

            the_FPA_esti = np.arctan2(disp_x[-1], disp_y[-1]) * 180 / np.pi
            the_sample = int((step[0] + step[1]) / 2)
            FPA_estis[the_sample] = the_FPA_esti

        return FPA_estis