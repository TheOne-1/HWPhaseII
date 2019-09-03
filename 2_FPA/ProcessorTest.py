import numpy as np
from transforms3d.euler import mat2euler, euler2mat
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from const import MOCAP_SAMPLE_RATE
from numpy.linalg import norm


def get_angles_via_acc_integration(data_df, IMU_location, euler_angles, l_strikes, l_offs):
    axis_name_acc = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
    acc_IMU = data_df[axis_name_acc].values
    acc_IMU = data_filt(acc_IMU, 6, MOCAP_SAMPLE_RATE)

    data_len = acc_IMU.shape[0]
    vel_foot = np.zeros([data_len, 3])
    pos_foot = np.zeros([data_len, 3])
    delta_t = 1 / MOCAP_SAMPLE_RATE
    FPA_angles = np.zeros([data_len])
    euler_angles = euler_angles / 180 * np.pi

    for l_off in l_offs:
        l_strike = l_strikes[l_strikes > l_off]
        l_strike = l_strike[l_strike < l_off + 200]
        if len(l_strike) == 1:      # swing_phase detected
            l_strike = l_strike[0]
            integration_start = l_off - 15
            integration_end = l_strike - 10
            for i_sample in range(integration_start, integration_end):
                # only roll and pitch
                dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
                acc_foot = acc_IMU[i_sample, :].T
                acc_foot = np.matmul(dcm_mat, acc_foot)
                vel_foot[i_sample] = acc_foot * delta_t + vel_foot[i_sample - 1]

            # vel_foot[integration_start:integration_end, :] = data_filt(vel_foot[integration_start:integration_end, :], 6, MOCAP_SAMPLE_RATE)
    vel_foot = data_filt(vel_foot, 6, MOCAP_SAMPLE_RATE)

    for l_off in l_offs:
        l_strike = l_strikes[l_strikes > l_off]
        l_strike = l_strike[l_strike < l_off + 200]
        if len(l_strike) == 1:      # swing_phase detected
            l_strike = l_strike[0]
            integration_start = l_off - 15
            integration_end = l_strike - 10
            for i_sample in range(integration_start, integration_end):
                pos_foot[i_sample] = (vel_foot[i_sample] + vel_foot[i_sample]) / 2 * delta_t + pos_foot[i_sample - 1]
            FPA_angles[integration_start:integration_end] = np.arctan2(pos_foot[integration_start:integration_end, 0],
                                                                       pos_foot[integration_start:integration_end, 1])
    return FPA_angles * 180 / np.pi - 180


def kalman_orientation(data_df, IMU_location, yaw_correction, l_strikes, l_offs):
    axis_name_acc = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
    acc_IMU = data_df[axis_name_acc].values
    acc_IMU = data_filt(acc_IMU, 10, MOCAP_SAMPLE_RATE)
    axis_name_gyr = [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
    gyr_IMU = data_df[axis_name_gyr].values
    gyr_IMU = data_filt(gyr_IMU, 10, MOCAP_SAMPLE_RATE)

    # find stance phase
    data_len = data_df.shape[0]
    stance_phase_flag = np.zeros([data_len], dtype=bool)
    for l_strike in l_strikes:
        l_off = l_offs[l_offs > l_strike]
        l_off = l_off[l_off < l_strike + 200]
        if len(l_off) == 1:      # stance phase detected
            stance_phase_flag[l_strike + 10:l_off[0] - 15] = True

    delta_t = 100 / MOCAP_SAMPLE_RATE
    euler_angles_esti = np.zeros([data_len, 3])
    for i_sample in range(data_len):
        euler_angles_esti[i_sample, :] = euler_angles_esti[i_sample-1, :] + delta_t * (gyr_IMU[i_sample, :] + gyr_IMU[i_sample, :]) / 2

        # # correct the yaw based on acc integration
        # if i_sample+15 in l_strikes:
        #     euler_angles_esti[i_sample, 2] = euler_angles_esti[i_sample, 2] + 0.4 * (yaw_correction[i_sample] - euler_angles_esti[i_sample, 2])

        # correct the roll and pitch based on the gravity
        if stance_phase_flag[i_sample] == True:
            pitch_correction = - np.arctan2(acc_IMU[i_sample, 0], norm(acc_IMU[i_sample])) * 180 / np.pi       # axis 1
            roll_correction = np.arctan2(acc_IMU[i_sample, 1], norm(acc_IMU[i_sample])) * 180 / np.pi      # axis 0
            euler_angles_esti[i_sample, 0] = euler_angles_esti[i_sample, 0] + 0.02 * (roll_correction - euler_angles_esti[i_sample, 0])
            euler_angles_esti[i_sample, 1] = euler_angles_esti[i_sample, 1] + 0.02 * (pitch_correction - euler_angles_esti[i_sample, 1])
    return euler_angles_esti


def get_mean_FPA(angles, l_strikes, l_offs, start_phase, end_phase):
    mean_FPAs = []
    for l_strike in l_strikes:
        l_off = l_offs[l_offs > l_strike]
        l_off = l_off[l_off < l_strike + 200]
        if len(l_off) == 1:      # stance phase detected
            l_off = l_off[0]
            start_sample = int(l_strike + start_phase * (l_off - l_strike))
            end_sample = int(l_strike + end_phase * (l_off - l_strike))
            mean_FPA = np.mean(angles[start_sample:end_sample])
            mean_FPAs.append(mean_FPA)
    return mean_FPAs


def get_angles_via_acc_ratio(data_df, IMU_location, euler_angles):
    axis_name_acc = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
    acc_IMU = data_df[axis_name_acc].values
    acc_IMU = data_filt(acc_IMU, 6, MOCAP_SAMPLE_RATE)

    data_len = acc_IMU.shape[0]
    angles = np.zeros([data_len])
    euler_angles = euler_angles / 180 * np.pi
    for i_sample in range(data_len):
        acc_foot = acc_IMU[i_sample, :]
        # dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
        # acc_foot = np.matmul(dcm_mat, acc_foot.T)
        angles[i_sample] = np.arctan2(acc_foot[0], abs(acc_foot[1])) * 180 / np.pi
    return angles


def get_angles_via_gyr_ratio(data_df, IMU_location):
    axis_name_gyr = [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
    gyr_IMU = data_df[axis_name_gyr].values
    gyr_IMU = data_filt(gyr_IMU, 6, MOCAP_SAMPLE_RATE)

    # plt.plot(gyr_IMU[:, 0])
    # plt.plot(gyr_IMU[:, 1])
    # plt.show()

    angles = np.arctan2(gyr_IMU[:, 1], abs(gyr_IMU[:, 0])) * 180 / np.pi
    return angles


def get_angles_via_gra_mag(data_df, IMU_location):
    axis_name_gravity = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
    data_gravity = data_df[axis_name_gravity].values

    axis_name_mag = [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
    data_mag = data_df[axis_name_mag].values

    fun_norm_vect = lambda v: v / np.linalg.norm(v)
    vector_2 = np.apply_along_axis(fun_norm_vect, 1, data_gravity)

    vector_0 = np.array(list(map(np.cross, data_mag, data_gravity)))
    vector_0 = np.apply_along_axis(fun_norm_vect, 1, vector_0)

    vector_1 = np.array(list(map(np.cross, vector_2, vector_0)))
    vector_1 = np.apply_along_axis(fun_norm_vect, 1, vector_1)

    dcm_mat = np.array([vector_0, vector_1, vector_2])
    dcm_mat = np.swapaxes(dcm_mat, 0, 1)
    euler_angles = np.array(list(map(mat2euler, dcm_mat))) / np.pi * 180

    return euler_angles


def get_real_euler_angles(data_df, IMU_location):
    angle_names = [IMU_location + '_' + axis for axis in ['roll', 'pitch', 'yaw']]
    real_angles = data_df[angle_names].values
    return real_angles


# def get_real_euler_angles(data_df, IMU_location):
#     angle_names = [IMU_location + '_' + axis for axis in ['roll', 'pitch', 'yaw']]
#     real_angles = data_df[angle_names].values
#     return real_angles


def compare_angles(angles_est, angles_mea, axis=0):
    plt.figure()
    plt.plot(angles_est[:, axis])
    plt.plot(angles_mea[:, axis])


def get_rotation_via_static_cali(data_df, IMU_location):
    axis_name_gravity = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
    data_gravity = data_df[axis_name_gravity]
    vector_gravity = np.mean(data_gravity.values, axis=0)

    axis_name_mag = [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
    data_mag = data_df[axis_name_mag]
    vector_mag = np.mean(data_mag.values, axis=0)

    vector_2 = vector_gravity / np.linalg.norm(vector_gravity)
    vector_0 = np.cross(vector_mag, vector_gravity)
    vector_0 = vector_0 / np.linalg.norm(vector_0)
    vector_1 = np.cross(vector_2, vector_0)
    vector_1 = vector_1 / np.linalg.norm(vector_1)

    dcm_mat = np.array([vector_0, vector_1, vector_2])
    return dcm_mat


def data_filt(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filt = filtfilt(b, a, data)
    else:
        data_filt = filtfilt(b, a, data, axis=0)
    return data_filt


def test_rota_mat(data_df, IMU_location, euler_angles):
    axis_name_acc = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
    acc_IMU = data_df[axis_name_acc].values
    acc_IMU = data_filt(acc_IMU, 6, MOCAP_SAMPLE_RATE)
    acc_foot = np.zeros(acc_IMU.shape)
    euler_angles = euler_angles / 180 * np.pi
    for i_sample in range(acc_foot.shape[0]):
        dcm_mat = euler2mat(euler_angles[i_sample, 0], euler_angles[i_sample, 1], 0)
        acc_foot[i_sample, :] = np.matmul(dcm_mat, acc_IMU[i_sample, :].T)
    plt.plot(acc_foot[:, 0])
    plt.plot(acc_foot[:, 1])
    plt.plot(acc_foot[:, 2])
    plt.show()






