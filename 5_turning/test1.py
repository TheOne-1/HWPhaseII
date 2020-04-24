import numpy as np
import matplotlib.pyplot as plt
from const import HAISHENG_SENSOR_SAMPLE_RATE, DATA_COLUMNS_IMU
import pandas as pd
from DataProcessorHS import HaishengSensorReaderNoInterpo
from DataProcessorHS import ParamInitializerHS
from ProcessorFPAHS import InitFPA
from ProcessorFPA import ProcessorFPA
from DataProcessorHS import StrikeOffDetectorIMU_HS_Overground
from numpy import cos, sin
import copy
from numpy.linalg import norm


DATA_PATH_TURNING = 'D:\Tian\Research\Projects\ML Project\data\\20200114FPA_ExperimentIIData\\200402TEST'
PLACEMENT_OFFSET = 0
offset_rad = - np.deg2rad(PLACEMENT_OFFSET)
placement_R_foot_sensor = np.array([
    [cos(offset_rad), sin(offset_rad), 0],
    [-sin(offset_rad), cos(offset_rad), 0],
    [0, 0, 1]])

def load_hs_sensor_data(data_path, placement_offset):
    sensor_data_reader = HaishengSensorReaderNoInterpo(data_path)
    sensor_data_cols = ['sample'] + DATA_COLUMNS_IMU + ['FPA_tbme_average', 'FPA_tbme_raw']
    sensor_data_all = sensor_data_reader.data_raw_df[sensor_data_cols]
    sensor_data_all[['FPA_tbme_raw', 'FPA_tbme_average']] -= placement_offset
    sensor_data_all[['acc_x', 'acc_y', 'acc_z']] = 9.81 * sensor_data_all[['acc_x', 'acc_y', 'acc_z']]
    sensor_data_all = sensor_data_all.rename(columns={'sample': 'IMU_frame'})
    return sensor_data_all

def find_walking_ends(data_df):
    """Find the walking end via a three second foot lift."""
    check_len = 2 * HAISHENG_SENSOR_SAMPLE_RATE     # 3 seconds
    skip_len = int(0.5 * HAISHENG_SENSOR_SAMPLE_RATE)       # 0.5 seconds

    data_len = data_df.shape[0]
    acc_data = data_df[['acc_x', 'acc_y', 'acc_z']].values
    acc_smoothed = np.zeros(acc_data.shape)
    acc_smoothed_y = acc_smoothed[:, 1]
    for i_axis in range(3):
        acc_smoothed[:, i_axis] = ProcessorFPA.smooth(acc_data[:, i_axis], 50, 'flat')

    walking_ends = []
    last_clip_is_end = False
    for i_pos in range(0, data_len, skip_len):
        if np.mean(acc_smoothed_y[i_pos:i_pos+check_len]) < -5:
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
                    if steps[j_step][1] - steps[j_step-1][1] < walking_bout_separation_criterion:
                        steps_of_trial.append(steps[j_step-1])
                    else:
                        break
                break
        steps_of_trials_dict[i_trial] = steps_of_trial
        i_trial -= 1
        current_i_step = j_step
    return steps_of_trials_dict


def get_FPA_tbme_step(gait_data_df, steps):
    FPA_tbmes = []
    # start from the second step, so that virtual step won't be calculated
    for i_step in range(1, len(steps)):
        current_step = steps[i_step]
        current_middle_sample = round((current_step[0] + current_step[1]) / 2)
        FPA_tbme = gait_data_df.loc[current_step[1], 'FPA_tbme_average']
        FPA_tbmes.append([FPA_tbme, current_middle_sample])
    return FPA_tbmes


def get_strike_off_from_imu(gait_data_df, param_data_df, trial_name, check_strike_off=True,
                            plot_the_strike_off=False):
    my_detector = StrikeOffDetectorIMU_HS_Overground(trial_name, gait_data_df, param_data_df, 'l_foot',
                                                     HAISHENG_SENSOR_SAMPLE_RATE)
    strike_delay, off_delay = -5, 0   # delay from the peak
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


for i_trial in range(6, 7):
    trial_path = DATA_PATH_TURNING + '\\DATA_' + str(i_trial) + '.CSV'

    data_df = load_hs_sensor_data(trial_path, placement_offset=PLACEMENT_OFFSET)
    data_df2 = load_hs_sensor_data(trial_path, placement_offset=0)

    walking_ends = find_walking_ends(data_df)

    estimated_strikes, estimated_offs = get_strike_off_from_imu(data_df, None, None, False)
    gait_param_df = pd.DataFrame(np.column_stack([estimated_strikes, estimated_offs]))
    gait_param_df.columns = ['strikes_IMU', 'offs_IMU']
    steps = ParamInitializerHS.get_legal_steps(estimated_strikes, estimated_offs, 'l', False)
    steps_of_trials_dict = get_steps_of_each_trial(walking_ends, steps)
    add_a_virtual_step(steps_of_trials_dict, estimated_offs)

    _, stance_phase_flag = InitFPA.initalize_steps_and_stance_phase(data_df, gait_param_df)
    euler_angles_esti = InitFPA.get_euler_angles_complementary_from_stance(
        data_df, stance_phase_flag)
    acc_IMU_rotated = InitFPA.get_rotated_acc(placement_R_foot_sensor, data_df, euler_angles_esti)
    acc_IMU_rotated2 = InitFPA.get_rotated_acc(np.eye(3), data_df, euler_angles_esti)

    plt.figure()
    plt.plot(-data_df['gyr_x'] * 10)

    for i_trial in range(len(walking_ends)):
        trial_steps = steps_of_trials_dict[i_trial]
        trial_steps.reverse()

        FPA_tbme = get_FPA_tbme_step(data_df, trial_steps)
        for fpa in FPA_tbme:
            plt.plot(fpa[1], fpa[0], 'c^')

        FPA_esti = InitFPA.get_FPA_via_max_acc(acc_IMU_rotated, trial_steps, start_percent=0.6, end_percent=1.2)
        for i_sample in range(len(FPA_esti)):
            if FPA_esti[i_sample] != 0:
                plt.plot(i_sample, FPA_esti[i_sample], 'bx')

        for step in trial_steps:
            plt.plot(step[0], 0, 'go')
            plt.plot(step[1], 0, 'rx')
    plt.show()















