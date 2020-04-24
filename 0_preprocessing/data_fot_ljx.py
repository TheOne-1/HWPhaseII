from const import COLUMN_FOR_HUAWEI_1000, PROCESSED_DATA_PATH, SUB_NAMES, RAW_DATA_PATH, LXJ_DATA_PATH, \
    TRUNK_SUBTRIAL_NAMES, TRIAL_NAMES, MOCAP_SAMPLE_RATE
import os
import pandas as pd
import xlrd
import numpy as np


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


def copy_200_data(trial_name, sub_data_path):
    # copy 200 Hz data
    gait_data_200_df = pd.read_csv(ori_200_path + '\\' + trial_name + '.csv', index_col=False)
    gait_data_200_df_ljx = gait_data_200_df[COLUMN_FOR_LJX]
    data_file_str = '{folder_path}\\{trial_name}.csv'.format(
        folder_path=sub_data_path, trial_name=trial_name)
    gait_data_200_df_ljx.to_csv(data_file_str, index=False)


for subject_folder in SUB_NAMES:
    print(subject_folder)
    processed_200_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\200Hz'
    ori_200_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\200Hz'
    data_path_standing, data_path_walking = initialize_path(LXJ_DATA_PATH, subject_folder)

    readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme_' + subject_folder + '.xlsx'

    # init static
    copy_200_data('static', data_path_standing)
    copy_200_data('static trunk', data_path_standing)

    # init walking trials
    for trial_name in TRIAL_NAMES[2:]:
        gait_data_200_df_ljx = copy_200_data(trial_name, data_path_walking)

