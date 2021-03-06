import numpy as np
import copy

TRIAL_NAMES = ['static', 'static trunk', 'baseline 10', 'FPA 10', 'trunk 10', 'baseline 12', 'FPA 12', 'trunk 12',
               'baseline 14', 'FPA 14', 'trunk 14']

SUB_AND_TRIALS = {'190803LiJiayi': TRIAL_NAMES, '190806SunDongxiao': TRIAL_NAMES, '190806WangDianxin': TRIAL_NAMES,
                  '190810LiuSensen': TRIAL_NAMES, '190813Caolinfeng': TRIAL_NAMES, '190813ZengJia': TRIAL_NAMES,
                  '190815WangHan': TRIAL_NAMES, '190815QiuYue': TRIAL_NAMES, '190824ZhangYaqian': TRIAL_NAMES,
                  '190816YangCan': TRIAL_NAMES, '190820FuZhenzhen': TRIAL_NAMES, '190820FuZhinan': TRIAL_NAMES,
                  '190822HeMing': TRIAL_NAMES[:3] + TRIAL_NAMES[4:],
                  '190826MouRongzi': TRIAL_NAMES, '190828LiangJie': TRIAL_NAMES,
                  '190829JiBin': TRIAL_NAMES, '190829ZhaoJiamin': TRIAL_NAMES, '190831XieJie': TRIAL_NAMES,
                  '190831GongChangyang': TRIAL_NAMES}

FOOT_SENSOR_BROKEN_SUBS = ('190826MouRongzi', '190828LiangJie', '190829JiBin', '190829ZhaoJiamin', '190831XieJie',
                           '190831GongChangyang')

SUB_NAMES = tuple(SUB_AND_TRIALS.keys())

SUB_AND_PARAM_TRIALS = copy.deepcopy(SUB_AND_TRIALS)  # trials for parameter calculation
for key in SUB_AND_PARAM_TRIALS.keys():
    if 'static' in SUB_AND_PARAM_TRIALS[key]:
        SUB_AND_PARAM_TRIALS[key].remove('static')

SUB_AND_WALKING_TRIALS = copy.deepcopy(SUB_AND_TRIALS)
for key in SUB_AND_WALKING_TRIALS.keys():
    if 'static' in SUB_AND_WALKING_TRIALS[key]:
        SUB_AND_WALKING_TRIALS[key].remove('static')
    if 'static trunk' in SUB_AND_WALKING_TRIALS[key]:
        SUB_AND_WALKING_TRIALS[key].remove('static trunk')

WALKING_TRIALS = SUB_AND_WALKING_TRIALS[SUB_NAMES[0]]

TRUNK_TRIALS = (TRIAL_NAMES[1], TRIAL_NAMES[4], TRIAL_NAMES[7], TRIAL_NAMES[10])
FPA_TRIALS = (TRIAL_NAMES[3], TRIAL_NAMES[6], TRIAL_NAMES[9])

# in Haisheng sensor's column names, x and y are switched to make it the same as Xsens column
COLUMN_NAMES_HAISHENG = ['hour', 'minute', 'second', 'millisecond', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y',
                         'gyr_z', 'mag_y', 'mag_x', 'mag_z']

SEGMENT_MARKERS = {'trunk': ['RAC', 'LAC', 'C7'], 'pelvis': ['RIAS', 'LIAS', 'LIPS', 'RIPS'],
                   'l_thigh': ['LTC1', 'LTC2', 'LTC3', 'LTC4', 'LFME', 'LFLE'],
                   'r_thigh': ['RTC1', 'RTC2', 'RTC3', 'RTC4', 'RFME', 'RFLE'],
                   'l_shank': ['LSC1', 'LSC2', 'LSC3', 'LSC4', 'LTAM', 'LFAL'],
                   'r_shank': ['RSC1', 'RSC2', 'RSC3', 'RSC4', 'RTAM', 'RFAL'],
                   'l_foot': ['LFM2', 'LFM5', 'LFCC'], 'r_foot': ['RFM2', 'RFM5', 'RFCC']}

FORCE_NAMES = ['marker_frame', 'f_1_x', 'f_1_y', 'f_1_z', 'c_1_x', 'c_1_y', 'c_1_z',
               'f_2_x', 'f_2_y', 'f_2_z', 'c_2_x', 'c_2_y', 'c_2_z']

DATA_COLUMNS_XSENS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z',
                      'roll', 'pitch', 'yaw']

DATA_COLUMNS_IMU = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']

XSENS_SENSOR_LOACTIONS = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']

XSENS_FILE_NAME_DIC = {'trunk': 'MT_0370064E_000.mtb', 'pelvis': 'MT_0370064C_000.mtb',
                       'l_thigh': 'MT_0370064B_000.mtb', 'l_shank': 'MT_0370064A_000.mtb',
                       'l_foot': 'MT_03700647_000.mtb'}

HAISHENG_SENSOR_SAMPLE_RATE = 100
MOCAP_SAMPLE_RATE = 200
PLATE_SAMPLE_RATE = 1000
STATIC_STANDING_PERIOD = 10  # unit: second

with open('..\\configuration.txt', 'r') as config:
    RAW_DATA_PATH = config.readline()

path_index = RAW_DATA_PATH.rfind('\\', 0, len(RAW_DATA_PATH) - 2)
PROCESSED_DATA_PATH = RAW_DATA_PATH[:path_index] + '\\ProcessedData'
HUAWEI_DATA_PATH = RAW_DATA_PATH[:path_index] + '\\DataForHuawei'
LXJ_DATA_PATH = RAW_DATA_PATH[:path_index] + '\\DataForLJX'

# COP_DIFFERENCE = np.array([279.4, 784, 0])  # reset coordinate difference

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'rosybrown', 'firebrick', 'olive', 'darkgreen',
          'slategray', 'navy', 'slateblue', 'm', 'indigo', 'maroon', 'peru', 'seagreen']

_10_TRIALS = ('baseline 10', 'FPA 10')
_12_TRIALS = ('baseline 12', 'FPA 12')
_14_TRIALS = ('baseline 14', 'FPA 14')

ROTATION_VIA_STATIC_CALIBRATION = False

TRIAL_START_BUFFER = 3  # 3 seconds filter buffer
FILTER_WIN_LEN = 100  # The length of FIR filter window

FONT_SIZE = 18
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'DejaVu Sans'}
FONT_DICT_LARGE = {'fontsize': 20, 'fontname': 'DejaVu Sans'}
FONT_DICT_SMALL = {'fontsize': 16, 'fontname': 'DejaVu Sans'}
FONT_DICT_X_SMALL = {'fontsize': 14, 'fontname': 'DejaVu Sans'}
LINE_WIDTH = 2

COLUMN_FOR_HUAWEI = ['marker_frame', 'f_1_x', 'f_1_y', 'f_1_z', 'c_1_x', 'c_1_y', 'c_1_z',
                     'C7_x', 'C7_y', 'C7_z', 'LIPS_x', 'LIPS_y', 'LIPS_z', 'RIPS_x', 'RIPS_y', 'RIPS_z',
                     'LFCC_x', 'LFCC_y', 'LFCC_z', 'LFM5_x', 'LFM5_y', 'LFM5_z', 'LFM2_x', 'LFM2_y', 'LFM2_z',
                     'RFCC_x', 'RFCC_y', 'RFCC_z', 'RFM5_x', 'RFM5_y', 'RFM5_z', 'RFM2_x', 'RFM2_y', 'RFM2_z',
                     'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z', 'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
                     'trunk_mag_x', 'trunk_mag_y', 'trunk_mag_z',
                     'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z', 'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
                     'l_foot_mag_x', 'l_foot_mag_y', 'l_foot_mag_z']

COLUMN_FOR_HUAWEI_1000 = ['marker_frame', 'f_1_x', 'f_1_y', 'f_1_z', 'c_1_x', 'c_1_y', 'c_1_z']

SPECIFIC_CALI_MATRIX = {}

TRUNK_SUBTRIAL_NAMES = ('normal trunk', 'large sway', 'backward inclination',
                        'backward TI large TS', 'forward inclination', 'forward TI large TS')

accuracy_path = 'D:/Tian/Research/Projects/HuaweiProject/codes/190807DataProcessPhaseII/2_FPA/'

FPA_NAME_LIST = ['fpa_vicon', 'fpa_tbme_no_cali', 'fpa_acc_ratio']

DATA_PATH_HS = 'D:/Tian/Research/Projects/ML Project/data/haisheng data/'

# s9_lyf's data is strange
SUB_AND_TRIALS_HS = {
    's5_ylw': [75, 74, 78, 77, 79, 76, 80],
    's10_gk': [27, 23, 22, 26, 21, 25, 24],
    's11_zy': [33, 30, 32, 29, 28, 34, 31],
    's12_yxc': [70, 68, 73, 71, 69, 72, 67],
    's13_xy': [53, 51, 57, 52, 55, 56, 54],
    's14_pbs': [85, 87, 89, 86, 88, 90, 91],
    's1_wy': [40, 45, 46, 44, 39, 43, 41],
    's9_lyf': [14, 13, 84, 83, 15, 18, 16],
    's2_xjk': [48, 47, 52, 51, 50, 53, 56],
    's3_js': [70, 68, 64, 62, 63, 61, 69],
    's7_xhs': [123, 124, 122, 116, 113, 115, 119],
    's8_jon': [125, 128, 135, 126, 127, 129, 131],
    # 's6_hyj': [99, 106, 105, 108, 98, 95, 97]     # vicon和sensor数据不对应
    # 's4_tt': [73, 82, 78, 79, 75, 77, 81],        # 跑步机力噪声过大
}

SUB_TRIAL_NUM = {
    's1_wy': [2, 6, 7, 5, 1, 4, 3],
    's2_xjk': [2, 1, 6, 5, 4, 7, 3],
    's3_js': [5, 6, 4, 2, 3, 1, 7],
    's9_lyf': [2, 1, 7, 5, 3, 6, 4],
    's7_xhs': [6, 7, 5, 3, 1, 2, 4],
    's8_jon': [1, 4, 7, 2, 3, 5, 6],
    # 's4_tt': [1, 7, 4, 5, 2, 3, 2],
    # 's6_hyj': [4, 7, 6, 5, 3, 1, 2],
}

VICON_FOLDER_2_SUB = ('s1_wy', 's2_xjk', 's3_js', 's4_tt', 's6_hyj', 's7_xhs', 's8_jon')

SUB_TRIAL_NAME_SPECIAL = {
    's9_lyf': {-1: 'lyf_l_34', 5: 'lyf_l_34_57', 7: 'lyf_l_34_57'},
    's7_xhs': {-1: 'xhs_l_34_5_6_7', 1: 'xhs_l_34_1', 2: 'xhs_l_34_2_3', 3: 'xhs_l_34_2_3', 4: 'xhs_l_34_4'},
    's8_jon': {-1: 'jon_l_34_1_2_3_4_5', 6: 'jon_l_34_6', 7: 'jon_l_34_7'}
}

SUB_NAMES_HS = tuple(SUB_AND_TRIALS_HS.keys())

MARKERS_HS = ['l_toe', 'l_heel']

SUB_SELECTED_SPEEDS = {'s5_ylw': 1.1, 's10_gk': 1.2, 's11_zy': 1.2,
                       's12_yxc': 1.1, 's13_xy': 1.2, 's14_pbs': 1.2,
                       's1_wy': 1.15, 's2_xjk': 1.2, 's3_js': 1.2,
                       's4_tt': 1.15, 's6_hyj': 1.1, 's7_xhs': 1.2,
                       's8_jon': 1.2, 's9_lyf': 1}

TRIAL_NAMES_HS = ['normal', 'small_in', 'medium_in', 'large_in', 'small_out', 'medium_out', 'large_out']

FPA_NAME_LIST_HS = ['FPA_true', 'FPA_estis', 'FPA_tbme']




