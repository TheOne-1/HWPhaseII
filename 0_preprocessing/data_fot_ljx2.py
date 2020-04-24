"""LJX wants the whole trial and orientation of the sensor"""

from const import PROCESSED_DATA_PATH, SUB_NAMES, RAW_DATA_PATH, LXJ_DATA_PATH, TRIAL_NAMES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from XsensReader import XsensReaderLjx


COLUMN_FOR_LJX = ['l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z', 'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
                  'l_thigh_mag_x', 'l_thigh_mag_y', 'l_thigh_mag_z']


def initialize_path(ljx_data_path, subject_folder):
    # create folder for this subject
    data_path_standing = ljx_data_path + '\\' + subject_folder + '\\standing'
    data_path_walking = ljx_data_path + '\\' + subject_folder + '\\walking'
    if not os.path.exists(data_path_standing):
        os.makedirs(data_path_standing)
    if not os.path.exists(data_path_walking):
        os.makedirs(data_path_walking)
    return data_path_standing, data_path_walking


def transfer_xsens_data(trial_name, sub_data_path):
    xsens_file = ori_200_path + '\\' + trial_name + '\\' + 'MT_0370064B_000.mtb'
    xsens_reader = XsensReaderLjx(xsens_file, None, 'l_thigh', trial_name)
    gait_data_df = xsens_reader.data_processed_df

    data_file_str = '{folder_path}\\{trial_name}.csv'.format(
        folder_path=sub_data_path, trial_name=trial_name)
    gait_data_df.to_csv(data_file_str, index=False)
    return gait_data_df


def save_start_end(start_end_df, sub_data_path):
    data_file_str = '{folder_path}\\start_end.csv'.format(folder_path=sub_data_path)
    start_end_df.columns = ['trial_name', 'start', 'end']
    start_end_df.to_csv(data_file_str, index=False)


def find_gait_start_end(gait_data_df, start_offset=600):
    gait_data_processed_df = pd.read_csv(processed_200_path + '\\' + trial_name + '.csv', index_col=False)
    start_values = gait_data_processed_df['l_thigh_acc_x'].iloc[0]
    end_values = gait_data_processed_df['l_thigh_acc_x'].iloc[-1]
    raw_df_values = gait_data_df['l_thigh_acc_x']
    start_loc = np.where(abs(raw_df_values - start_values) < 1e-7)[0][0]
    end_loc = np.where(abs(raw_df_values - end_values) < 1e-7)[0][0]
    start_loc += start_offset
    return start_loc, end_loc


for subject_folder in SUB_NAMES[:12] + SUB_NAMES[13:]:
    print(subject_folder)
    processed_200_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\200Hz'
    ori_200_path = RAW_DATA_PATH + '\\' + subject_folder + '\\xsens'
    data_path_standing, data_path_walking = initialize_path(LXJ_DATA_PATH, subject_folder)

    # init static
    transfer_xsens_data('static', data_path_standing)
    transfer_xsens_data('static trunk', data_path_standing)

    walking_start_end_df = pd.DataFrame()
    for trial_name in TRIAL_NAMES[2:]:
        gait_data_df = transfer_xsens_data(trial_name, data_path_walking)
        start_loc, end_loc = find_gait_start_end(gait_data_df)
        current_start_end_df = pd.DataFrame([[trial_name, start_loc, end_loc]])
        walking_start_end_df = pd.concat([walking_start_end_df, current_start_end_df])

        # plt.plot(gait_data_df['l_thigh_acc_x'])
        # plt.plot(start_loc, 0, '*')
        # plt.plot(end_loc, 0, '*')
        # plt.show()

    save_start_end(walking_start_end_df, data_path_walking)






























