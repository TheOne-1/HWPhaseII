from const import WALKING_TRIALS, TRIAL_NAMES, PROCESSED_DATA_PATH
import pandas as pd
import numpy as np
from ProcessorTest import *


sub = '190810LiuSensen'
trial = TRIAL_NAMES[3]

data_file = PROCESSED_DATA_PATH + '\\' + sub + '\\' + '200Hz\\' + trial + '.csv'
data_df = pd.read_csv(data_file, index_col=False)

param_file = PROCESSED_DATA_PATH + '\\' + sub + '\\' + '200Hz\\param_of_' + trial + '.csv'
param_df = pd.read_csv(param_file, index_col=False)

l_strikes = np.where(param_df['l_strikes'] == 1)[0]
l_offs = np.where(param_df['l_offs'] == 1)[0]

angles_esti = kalman_orientation(data_df, 'l_foot', None, l_strikes, l_offs)
angles_mea = get_real_euler_angles(data_df, 'l_foot')
angles_est_gra = get_angles_via_gra_mag(data_df, 'l_foot')
# yaw_correction = get_angles_via_acc_integration(data_df, 'l_foot', angles_mea, l_strikes, l_offs)

yaw_correction = get_angles_via_acc_ratio(data_df, 'l_foot', angles_esti)
fpa_est = get_mean_FPA(yaw_correction, l_strikes, l_offs, 0.95, 0.99)

# yaw_correction = get_angles_via_gyr_ratio(data_df, 'l_foot')
# fpa_est = get_mean_FPA(yaw_correction, l_strikes, l_offs, 0, 0.04)

fpa_true = get_mean_FPA(param_df['l_FPA'].values, l_strikes, l_offs, 0.2, 0.8)
from scipy.stats import pearsonr
corr = pearsonr(fpa_est, fpa_true)
# fpa_est = [angle / 2 for angle in fpa_est]
plt.figure()
plt.plot(fpa_est, fpa_true, '.')
plt.plot([-5, 40], [-5, 40], 'r')
plt.title(corr[0])
plt.show()


# plt.figure()
# acc_z = data_df['l_foot_acc_z'].values
# plt.plot(yaw_correction)
# plt.plot(param_df['l_FPA'])
# plt.plot(l_offs + 15, yaw_correction[l_offs + 15], 'r^')
# plt.plot(l_strikes + 2, yaw_correction[l_strikes + 2], 'g^')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(angles_est[:, 2])
# plt.plot(angles_mea[:, 2])
# plt.plot(param_df['l_FPA'] - 180)

# plt.figure()
# angle_diff = angles_est[:, 2] - param_df['l_FPA'].values + 180
# plt.plot(angle_diff)
# plt.grid()
# plt.plot(l_strikes - 15, angle_diff[l_strikes - 15], 'g^')

# for axis in range(3):
#     compare_angles(angles_est, angles_mea, axis)
# plt.show()

