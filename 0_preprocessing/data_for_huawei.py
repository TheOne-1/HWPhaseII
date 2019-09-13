from const import COLUMN_FOR_HUAWEI_1000, PROCESSED_DATA_PATH, SUB_NAMES, RAW_DATA_PATH, HUAWEI_DATA_PATH,\
    COLUMN_FOR_HUAWEI, FPA_TRIALS, TRUNK_SUBTRIAL_NAMES, TRIAL_NAMES, MOCAP_SAMPLE_RATE
import os
import pandas as pd
import xlrd


def initialize_path(huawei_data_path, subject_folder):
    # create folder for this subject
    fre_200_path = huawei_data_path + '\\' + subject_folder + '\\200Hz'
    fre_1000_path = huawei_data_path + '\\' + subject_folder + '\\1000Hz'
    if not os.path.exists(huawei_data_path + '\\' + subject_folder):
        os.makedirs(huawei_data_path + '\\' + subject_folder)
    if not os.path.exists(fre_200_path):
        os.makedirs(fre_200_path)
    if not os.path.exists(fre_1000_path):
        os.makedirs(fre_1000_path)
    return fre_200_path, fre_1000_path


def copy_200_data(trial_name):
    # copy 200 Hz data
    gait_data_200_df = pd.read_csv(ori_200_path + '\\' + trial_name + '.csv', index_col=False)
    gait_data_200_df_hw = gait_data_200_df[COLUMN_FOR_HUAWEI]
    data_file_str = '{folder_path}\\{trial_name}.csv'.format(
        folder_path=fre_200_path, trial_name=trial_name)
    gait_data_200_df_hw.to_csv(data_file_str, index=False)


def copy_param_data(trial_name):
    param_data_200_df = pd.read_csv(ori_200_path + '\\param_of_' + trial_name + '.csv', index_col=False)
    data_file_str = '{folder_path}\\param_of_{trial_name}.csv'.format(
        folder_path=fre_200_path, trial_name=trial_name)
    param_data_200_df.to_csv(data_file_str, index=False)


def copy_1000_data(trial_name):
    # copy 1000 Hz data
    gait_data_1000_df = pd.read_csv(ori_1000_path + '\\' + trial_name + '.csv', index_col=False)
    gait_data_1000_df_hw = gait_data_1000_df[COLUMN_FOR_HUAWEI_1000]
    data_file_str = '{folder_path}\\{trial_name}.csv'.format(
        folder_path=fre_1000_path, trial_name=trial_name)
    gait_data_1000_df_hw.to_csv(data_file_str, index=False)


def split_subtrial(trial_name, subtrials_to_keep, sensor_sampling_fre=MOCAP_SAMPLE_RATE):
    trunk_subtrial_names = TRUNK_SUBTRIAL_NAMES

    gait_data_200_df = pd.read_csv(ori_200_path + '\\' + trial_name + '.csv', index_col=False)
    param_data_200_df = pd.read_csv(ori_200_path + '\\param_of_' + trial_name + '.csv', index_col=False)
    gait_data_1000_df = pd.read_csv(ori_1000_path + '\\' + trial_name + '.csv', index_col=False)

    readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
    trial_id = TRIAL_NAMES.index(trial_name)
    if 'trunk' in trial_name:
        subtrial_ends = readme_sheet.row_values(trial_id + 2)[6:12]
    else:
        raise ValueError('Only split trunk trials')
    subtrial_ends = [int(element / MOCAP_SAMPLE_RATE * sensor_sampling_fre) for element in subtrial_ends]
    subtrial_duration = sensor_sampling_fre * 60
    for subtrial_id in subtrials_to_keep:
        subtrial_end = subtrial_ends[subtrial_id]
        subtrial_start = subtrial_end - subtrial_duration
        subtrial_name = trunk_subtrial_names[subtrial_id] + trial_name[-3:]

        gait_data_200_df_hw = gait_data_200_df[COLUMN_FOR_HUAWEI]
        gait_data_200_df_hw = gait_data_200_df_hw.iloc[subtrial_start:subtrial_end]
        data_file_str = '{folder_path}\\{subtrial_name}.csv'.format(
            folder_path=fre_200_path, subtrial_name=subtrial_name)
        gait_data_200_df_hw.to_csv(data_file_str, index=False)

        param_data_200_df_hw = param_data_200_df.iloc[subtrial_start:subtrial_end]
        data_file_str = '{folder_path}\\param_of_{trial_name}.csv'.format(
            folder_path=fre_200_path, trial_name=subtrial_name)
        param_data_200_df_hw.to_csv(data_file_str, index=False)

        gait_data_1000_df_hw = gait_data_1000_df[COLUMN_FOR_HUAWEI_1000]
        gait_data_1000_df_hw = gait_data_1000_df_hw.iloc[subtrial_start:subtrial_end]
        data_file_str = '{folder_path}\\{trial_name}.csv'.format(
            folder_path=fre_1000_path, trial_name=subtrial_name)
        gait_data_1000_df_hw.to_csv(data_file_str, index=False)


for subject_folder in SUB_NAMES:
    print(subject_folder)
    ori_200_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\200Hz'
    ori_1000_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\1000Hz'
    fre_200_path, fre_1000_path = initialize_path(HUAWEI_DATA_PATH, subject_folder)

    readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme_' + subject_folder + '.xlsx'

    # init static
    copy_200_data('static')
    copy_1000_data('static')
    copy_200_data('static trunk')
    copy_param_data('static trunk')
    copy_1000_data('static trunk')

    for trial_name in [TRIAL_NAMES[4], TRIAL_NAMES[7], TRIAL_NAMES[10]]:
        split_subtrial(trial_name, [0, 2, 4])

    # init FPA trials
    for trial_name in FPA_TRIALS:
        copy_200_data(trial_name)
        copy_param_data(trial_name)
        copy_1000_data(trial_name)






















