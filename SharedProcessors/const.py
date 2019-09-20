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
COLUMN_NAMES_HAISHENG = ['hour', 'minute', 'second', 'millisecond', 'acc_y', 'acc_x', 'acc_z', 'gyr_y', 'gyr_x',
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
