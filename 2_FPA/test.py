import numpy as np
from transforms3d.euler import euler2mat
from const import FILTER_WIN_LEN, MOCAP_SAMPLE_RATE, SUB_NAMES, PROCESSED_DATA_PATH, FPA_TRIALS
import pandas as pd
from numpy.linalg import norm
from transforms3d.euler import euler2mat
from ProcessorFPA import ProcessorFPA
import matplotlib.pyplot as plt


static_data_path = PROCESSED_DATA_PATH + '\\' + SUB_NAMES[1] + '\\200Hz\\static.csv'
static_data = pd.read_csv(static_data_path, index_col=False)

data_len = static_data.shape[0]

acc_IMU = static_data[['l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z']].values
euler_angles_meas = static_data[['l_foot_roll', 'l_foot_pitch', 'l_foot_yaw']].values

euler_angles_esti = np.zeros([data_len, 3])
acc_IMU_norm = norm(acc_IMU, axis=1)

roll_correction = np.arctan2(acc_IMU[:, 1], acc_IMU[:, 2])  # axis 0
pitch_correction = np.arctan2(-acc_IMU[:, 0], np.sqrt(acc_IMU[:, 1] ** 2 + acc_IMU[:, 2] ** 2))

euler_angles_esti[:, 0] = np.rad2deg(roll_correction)
euler_angles_esti[:, 1] = np.rad2deg(pitch_correction)

mean_esti = np.mean(euler_angles_esti, axis=0)
mean_meas = np.mean(euler_angles_meas, axis=0)
print('Orientation difference: ', end='')
print(mean_esti - mean_meas)

the_radians = np.deg2rad(mean_esti)
mat_meas_0 = euler2mat(-the_radians[2], -the_radians[1], -the_radians[0], 'szyx')
print('zyx: ', end='')
print(np.matmul(mat_meas_0, acc_IMU[0, :].T))

mat_meas_1 = euler2mat(the_radians[0], the_radians[1], the_radians[2], 'sxyz')
print('xyz: ', end='')
print(np.matmul(mat_meas_1, acc_IMU[0, :].T))

# test = {'190813Caolinfeng': FPA_TRIALS[:1]}
# FPA_processor = ProcessorFPA({}, test, 200, 'l', 'FPA_steps', 'l_foot', data_type=1,
#                              do_input_norm=False, do_output_norm=False)
#
# data_path = PROCESSED_DATA_PATH + '\\190813Caolinfeng\\200Hz\\FPA 10.csv'
# data = pd.read_csv(data_path, index_col=False)
# data_len = data.shape[0]
#
# euler_angles_meas = data[['l_foot_roll', 'l_foot_pitch', 'l_foot_yaw']].values
# plt.figure()
# plt.plot(euler_angles_meas[:, 0])
#
# FPA_processor.prepare_train_test(trial_ids=[3, 6, 9])
#
# plt.show()



















